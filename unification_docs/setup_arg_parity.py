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
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OLD = REPO / "canvit_pretrain" / "train" / "loop.py"
NEW = REPO / "canvit_pretrain" / "tasks" / "distill" / "task.py"

# Shared setup helpers whose arguments must agree between the two stacks.
TRACKED = ("init_normalizer_stats_from_tar", "init_normalizer_stats_from_tar_raw")

# The two stacks bind the same values to different local names. Normalizing these is
# what lets a textual comparison be meaningful; anything NOT listed here compares raw,
# so a genuinely different expression (`x` vs `x or 512`) still shows up as a diff.
ALIASES = {
    "self.cfg": "cfg",
    "self._device": "cfg.device",
    "self.scene_norm": "scene_norm",
    "self.cls_norm": "cls_norm",
    "train.first_shard_path()": "train_loader.first_shard_path()",
    "sz": "scene_size",
    "self._teacher_targets": "compute_raw_targets",
}


def _norm(expr: str) -> str:
    for a, b in ALIASES.items():
        expr = expr.replace(a, b)
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


def compare(old_src: Path = OLD, new_src: Path = NEW) -> list[str]:
    """Return a list of mismatch descriptions (empty == parity)."""
    old, new = calls(old_src), calls(new_src)
    problems: list[str] = []
    for fn in TRACKED:
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
                f"    only in {new_src.name}: {only_new}")
    return problems


def main() -> int:
    problems = compare()
    if not problems:
        print(f"setup-arg parity OK: {', '.join(TRACKED)} called identically by both stacks")
        return 0
    print("SETUP-ARG PARITY FAILED:", file=sys.stderr)
    for p in problems:
        print("  - " + p, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
