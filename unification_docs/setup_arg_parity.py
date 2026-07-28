"""Setup-argument parity: both stacks must call SHARED setup helpers the same way.

The exp23 foveated regression was not a bug in any function — it was a bug in an
*argument*. `train/loop.py` passed `cfg.normalizer_max_samples`; the harness passed
`self.cfg.normalizer_max_samples or 512`, which clobbered the documented `0` = "use the
whole shard" sentinel and gave the two stacks different target statistics off the same
shard. Both called the identical function, so every step-level parity check agreed and
the difference only surfaced ~24k steps into a 12-hour production run.

Step-level parity (parity_probe / parity_configs / harness_realdata_ab) compares the
per-step rollout from the SAME model state and batch — it cannot see a setup call that
built that state differently. This closes that blind spot statically: parse both call
sites, normalize the local aliases, and require the argument expressions to match.

No torch, no data, no GPU — pure AST, milliseconds.

Run:  .venv-cu126/bin/python unification_docs/setup_arg_parity.py
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OLD = REPO / "canvit_pretrain" / "train" / "loop.py"
# The harness spreads distill setup across the task and the neutral runner, so the "new"
# side is a SET of files: a call moving between them must not read as a missing call.
NEW = REPO / "canvit_pretrain" / "tasks" / "distill" / "task.py"
NEW_FILES = (NEW, REPO / "canvit_pretrain" / "harness" / "run.py")

# Shared setup helpers whose arguments must agree between the two stacks. Every one of
# these is imported by train/loop.py AND reachable from the harness; the exp23 bug was an
# argument to the first of them, so the whole surface is checked rather than that one.
TRACKED = (
    "init_normalizer_stats_from_tar",
    "init_normalizer_stats_from_tar_raw",
    "create_loaders",
    "create_model",
    "load_student_backbone",
    "load_teacher",
    "load_probe",
    "scene_size_px",
    # Called by train/loop.py under cfg.compile; the harness compiles only the student.
    # Tracked so the ASYMMETRY is reported rather than silently tolerated.
    "compile_teacher",
)

# Helpers the harness is KNOWN not to call, with the reason. Listing one here turns a
# hard failure into a reported divergence — it must be a decision, never an oversight.
KNOWN_ABSENT: dict[str, str] = {}

# The two stacks bind the same values to different local names. Normalizing these is
# what lets a textual comparison be meaningful; anything NOT listed here compares raw,
# so a genuinely different expression (`x` vs `x or 512`) still shows up as a diff.
ALIASES = {
    "self.cfg": "cfg",
    "self._device": "cfg.device",
    "self.scene_norm": "scene_norm",
    "self.cls_norm": "cls_norm",
    "train.normalizer_shard_paths": "train_loader.normalizer_shard_paths",
    "sz": "scene_size",
    "self._teacher_targets": "compute_raw_targets",
    # DDP identity: the old loop asks the ddp module, the harness receives it as a
    # parameter from run(). Same value, different plumbing.
    "ddp.rank()": "rank",
    "ddp.world_size()": "world_size",
    # Shard-schedule position. loop.py:368 computes exactly this expression into a local
    # on the webdataset path (`start_step = start_job_index * cfg.steps_per_job`).
    "self._start_job_index * self.cfg.steps_per_job": "start_step",
    "self._start_job_index": "start_job_index",
    # Locals bound to the same thing a line earlier.
    "backbone": "student_backbone",
    "device": "cfg.device",
    "self.cfg.canvas_patch_grid_size": "G",
    # Both now read the TEACHER's geometry/width (see distill/task.py) rather than the
    # student's or the config placeholder.
    "teacher_dim": "teacher.embed_dim",
    "teacher_patch": "patch_size",
    "self._scene_size_px()": "scene_size",
    "self._teacher": "teacher",
}


def _norm(expr: str) -> str:
    # SINGLE pass with word boundaries. A naive per-alias str.replace cascades
    # (`backbone`->`student_backbone` then matches again -> `student_student_backbone`)
    # and matches inside longer identifiers (`device` inside `cfg.device`), which
    # manufactures mismatches that do not exist.
    if ALIASES:
        pattern = re.compile("|".join(re.escape(k) for k in
                                      sorted(ALIASES, key=len, reverse=True)))

        def _sub(m: re.Match) -> str:
            start, end = m.start(), m.end()
            before = expr[start - 1] if start else ""
            after = expr[end] if end < len(expr) else ""
            # Don't fire inside a longer identifier/attribute chain. NB `before` is ""
            # at position 0 and `"" in "_."` is True in Python, so guard on truthiness
            # first or every match at the start of an expression is skipped.
            if (before and (before.isalnum() or before in "_.")) or after.isalnum() or after == "_":
                return m.group(0)
            return ALIASES[m.group(0)]

        expr = pattern.sub(_sub, expr)
    # lambdas differ in parameter name / call shape but express "teacher features for
    # these images"; compare the callee only.
    if expr.startswith("lambda"):
        expr = "lambda:" + ("compute_raw_targets" if "compute_raw_targets" in expr else expr)
    return " ".join(expr.split())


def calls(path: Path) -> dict[str, dict[str, str]]:
    """{function_name: {arg_key: normalized_expr}} for every tracked call in `path`."""
    tree = ast.parse(path.read_text())
    found: dict[str, dict[str, str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
        if fn not in TRACKED:
            continue
        args = {f"#{i}": _norm(ast.unparse(a)) for i, a in enumerate(node.args)}
        args |= {kw.arg: _norm(ast.unparse(kw.value)) for kw in node.keywords}
        found[fn] = args
    return found


def compare(old_src: Path = OLD, new_src: Path | tuple[Path, ...] = None,
            tracked: tuple[str, ...] | None = None,
            aliases: dict[str, str] | None = None,
            known_absent: dict[str, str] | None = None) -> list[str]:
    """Return a list of mismatch descriptions (empty == parity)."""
    global ALIASES, TRACKED, KNOWN_ABSENT
    saved = (ALIASES, TRACKED, KNOWN_ABSENT)
    if aliases is not None:
        ALIASES = aliases
    if tracked is not None:
        TRACKED = tracked
    if known_absent is not None:
        KNOWN_ABSENT = known_absent
    try:
        return _compare(old_src, new_src)
    finally:
        ALIASES, TRACKED, KNOWN_ABSENT = saved


def _compare(old_src: Path, new_src: Path | tuple[Path, ...] | None) -> list[str]:
    new_srcs = (new_src,) if isinstance(new_src, Path) else (new_src or NEW_FILES)
    old = calls(old_src)
    new: dict[str, dict[str, str]] = {}
    for f in new_srcs:
        if f.exists():
            new |= calls(f)
    new_name = "+".join(f.name for f in new_srcs)
    problems: list[str] = []
    for fn in TRACKED:
        if fn in old and fn not in new:
            if fn in KNOWN_ABSENT:
                continue  # a recorded decision, not an oversight
            problems.append(
                f"{fn}: called by {old_src.name} but NOT by the harness ({new_name}). "
                f"Either wire it up or record it in KNOWN_ABSENT with a reason.")
            continue
        if fn not in old or fn not in new:
            problems.append(f"{fn}: called in {'old' if fn in old else 'new'} stack only "
                            f"(old={fn in old} new={fn in new})")
            continue
        o, n = old[fn], new[fn]
        # Positional/keyword style differs between the stacks; compare the VALUES passed,
        # which is what actually reaches the function.
        ovals, nvals = sorted(o.values()), sorted(n.values())
        if ovals != nvals:
            only_old = [v for v in ovals if v not in nvals]
            only_new = [v for v in nvals if v not in ovals]
            problems.append(
                f"{fn}: argument mismatch\n"
                f"    only in {old_src.name}: {only_old}\n"
                f"    only in {new_name}: {only_new}")
    return problems


# --- the other two stack pairs -------------------------------------------------------
# Same question for the downstream tasks: `canvit_pretrain/{ade20k,in1k}/train.py` are the
# standalone trainers the harness tasks are supposed to reproduce. Only helpers BOTH sides
# call are compared — a helper the harness replaced wholesale (e.g. in1k's
# `rollout_cls_tokens`, superseded by harness/rollout.py::run_rollout) is a design change,
# not an argument bug, and is deliberately out of scope here.
P = REPO / "canvit_pretrain"

ADE_TRACKED = (
    "make_ade20k_loaders", "make_optimizer_and_scheduler", "make_random_viewpoints",
    "rollout_canvas_hidden", "eval_probe_on_batch", "ce_loss", "consumes_full_image",
    "from_pretrained_with_new_probe",
)
ADE_ALIASES = {
    "self.cfg": "cfg", "self._device": "cfg.device", "self._num_classes()": "NUM_CLASSES",
    "seg": "model", "self._model": "model",
    # locals bound to the same value
    "T": "cfg.n_timesteps", "is_fov": "is_foveated", "cg": "canvas_grid",
    "ious[t]": "val_iou[t]",
    "self._logits(readout)": "logits", "self.masks": "masks",
}
ADE_KNOWN_ABSENT = {
    "make_optimizer_and_scheduler":
        "By design: the harness builds optimizers from TrainSpec (harness/optim.py). "
        "Numerical equivalence to specialize's WarmupOneCycleLR is proven separately by "
        "harness/tests/test_optim.py::test_onecycle_matches_ade20k_reference_scheduler.",
}

IN1K_TRACKED = (
    "make_train_loader", "make_val_loader", "build_classifier", "consumes_full_image",
    "from_pretrained_with_new_head",
)
IN1K_ALIASES = {
    "self.cfg": "cfg", "self._device": "cfg.device", "clf": "model", "self._model": "model",
    "ddp.rank()": "rank", "ddp.world_size()": "world_size",
    # The standalone is single-job by construction — its own comment (in1k/train.py:160-161):
    # "job_index=0 and the shard slice spans the whole run - steps_per_job = max_steps,
    # NOT cfg.steps_per_job (which is a harness knob)". The harness adds SLURM-array
    # resume, so these two differ deliberately.
    "self._start_job_index": "0", "self._steps_per_job": "cfg.max_steps",
}
IN1K_KNOWN_ABSENT = {
    "from_pretrained_with_new_head":
        "Not a divergence: BOTH stacks construct the head via the shared "
        "in1k.train.build_classifier, which dispatches to from_pretrained_with_probe "
        "(finetune) or from_pretrained_with_new_head (frozen). The name only appears in "
        "train.py because build_classifier LIVES there.",
}

PAIRS: dict[str, tuple] = {
    "distill": (OLD, NEW_FILES, TRACKED, ALIASES, KNOWN_ABSENT),
    "ade20k": (P / "ade20k" / "train.py", (P / "tasks" / "ade20k" / "task.py",),
               ADE_TRACKED, ADE_ALIASES, ADE_KNOWN_ABSENT),
    "in1k": (P / "in1k" / "train.py", (P / "tasks" / "in1k" / "task.py",),
             IN1K_TRACKED, IN1K_ALIASES, IN1K_KNOWN_ABSENT),
}


def compare_all() -> dict[str, list[str]]:
    return {name: compare(old, new, tracked=t, aliases=a, known_absent=k)
            for name, (old, new, t, a, k) in PAIRS.items()}


def main() -> int:
    results = compare_all()
    rc = 0
    for name, problems in results.items():
        if not problems:
            print(f"  OK    {name:8s} — shared setup helpers called identically by both stacks")
            continue
        rc = 1
        print(f"  FAIL  {name}:", file=sys.stderr)
        for p in problems:
            print("    - " + p, file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
