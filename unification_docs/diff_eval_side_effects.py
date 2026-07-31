"""Does the harness's PERIODIC EVAL perturb training? The last untested stage.

Everything else in the pipeline is measured identical (doc 15 §A5.7), so the only place a
harness-vs-rl_train difference could still hide is state that the periodic eval leaves behind
between training steps.

An eval can reach training exactly two ways:
  (a) CONSUMING RNG — shifts the data order / eps-greedy stream. Different trajectory, SAME
      distribution: that is a seed change, not a bias, and cannot make results systematically
      worse.
  (b) MUTATING STATE — leaving a module in the wrong train/eval mode, moving the scorer's or
      probe's BatchNorm running stats, or dirtying the StateEncoder's ent_delta memory. This
      one CAN bias every subsequent step.

So detecting "a difference" is not enough; the two have to be told apart. This runs the same
training steps twice from one init on one batch stream — once with eval interleaved, once
without — and reports weights, BN buffers, and the chosen-glimpse sequence separately.

Usage: python diff_eval_side_effects.py [--steps 60] [--eval-every 20] [--val-batches 3]
"""
import argparse
import copy

import torch

from canvit_train.ade20k.config import Ade20kConfig
from canvit_train.ade20k.data import make_ade20k_loaders
from canvit_train.ade20k.task import POLICY_FEATURE_GROUPS, Ade20kRunTask
from canvit_train.harness.config import JointPolicyConfig
from canvit_train.harness.loop import apply_requires_grad
from canvit_train.harness.optim import build_optimizer_and_scheduler
from canvit_train.harness.rollout import run_rollout
from canvit_train.harness.spec import GroupOptim, ScheduleSpec, TrainSpec, fixed_horizon_bptt

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=60)
ap.add_argument("--eval-every", type=int, default=20)
ap.add_argument("--val-batches", type=int, default=3, help="0 = full val (slow)")
args = ap.parse_args()

T, dev = 5, torch.device("cuda")
torch.manual_seed(0)

cfg = Ade20kConfig(
    resize_mode="squish", scene_size=512, canvas_grid=64, augment=False, mode="frozen",
    probe_repo="canvit/probe-ade20k-40k-s512-c64-in21k", n_timesteps=T, eval_policy="policy",
    batch_size=16, num_workers=2, eval_batch_size=32,
    limit_val_batches=args.val_batches or None)
task = Ade20kRunTask(cfg)
task.rl = JointPolicyConfig(use_rl=True, feature_groups=POLICY_FEATURE_GROUPS,
                            prime_on_policy=0.5, select_bn_eval=True)
model, head = task.build_model(dev, prior_model_config=None)
cg = task.canvas_grid(model)
spec = TrainSpec.policy_only(freeze_model=True, bptt=fixed_horizon_bptt(frozen=True, horizon=T))
joint = task.build_policy(model, device=dev, canvas_grid=cg,
                          generator=torch.Generator(device=dev).manual_seed(0))
apply_requires_grad(model=model, head=head, joint=joint, spec=spec)
model.eval()
joint.scorer.train()
scorer = joint.scorer
init = copy.deepcopy(scorer.state_dict())

train_loader, val_loader = make_ade20k_loaders(cfg)
it = iter(train_loader)
batches = [next(it) for _ in range(args.steps)]
amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
gopt = GroupOptim(lr=2e-4, weight_decay=1e-2, betas=(0.9, 0.95),
                  schedule=ScheduleSpec(kind="warmup_constant", warmup_steps=1000))


def run(with_eval: bool):
    """The harness loop's ordering: eval BEFORE the update at step boundaries (loop.py)."""
    scorer.load_state_dict(init)
    joint.running.clear()
    joint.policy_selector.generator = torch.Generator(device=dev).manual_seed(4242)
    opt, sched = build_optimizer_and_scheduler(
        TrainSpec.policy_only(freeze_model=True,
                              bptt=fixed_horizon_bptt(frozen=True, horizon=T),
                              optim={"policy": gopt}),
        {"policy": list(scorer.parameters())})
    idx_trace, n_evals = [], 0
    for step in range(args.steps):
        if with_eval and args.eval_every and step % args.eval_every == 0:
            task.evaluate(model=model, head=head, val_loader=val_loader, device=dev,
                          step=step, joint=joint)
            n_evals += 1
        images, masks = batches[step]
        images, masks = images.to(dev), masks.to(dev)
        bound = task.bind((images.cpu(), masks.cpu()), dev, model=model, head=head)
        opt.zero_grad(set_to_none=True)
        run_rollout(model=model, images=images, task=bound, selector=joint.random_selector,
                    bptt=spec.bptt, branches=task.branches(), canvas_grid_size=cg,
                    amp_ctx=amp_ctx, task_weight=spec.task_weight, joint=joint)
        idx_trace.append(joint.policy_selector.last_aux["flat_idx"].clone())
        torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.0)
        opt.step(); sched.step()
    w = torch.cat([p.detach().flatten() for _, p in sorted(scorer.named_parameters())]).clone()
    bn = torch.cat([b.flatten().float() for n, b in sorted(scorer.named_buffers())
                    if "running" in n or "num_batches" in n]).clone()
    return w, bn, idx_trace, n_evals


wE, bnE, idxE, nev = run(True)
wN, bnN, idxN, _ = run(False)
# CONTROL: the no-eval arm against ITSELF. Without this the with-vs-without difference is
# uninterpretable — GPU kernel selection is not bit-reproducible across repeats, and that
# floor amplifies over steps. Any A-vs-B number must be read against this.
wC, bnC, idxC, _ = run(False)
ctl = ((wN - wC).norm() / wN.norm()).item()
ctl_flips = sum(int((a != b).sum()) for a, b in zip(idxN, idxC))

dw = (wE - wN).abs().max().item()
rel = ((wE - wN).norm() / wN.norm()).item()
dbn = (bnE - bnN).abs().max().item()
flips = [int((a != b).sum()) for a, b in zip(idxE, idxN)]
first = next((i for i, f in enumerate(flips) if f), None)

print(f"\nsteps={args.steps} eval_every={args.eval_every} ({nev} evals) "
      f"val_batches={args.val_batches or 'full'}")
print(f"  final weights      max|d|={dw:.3e}  relL2={rel:.3e}")
print(f"  scorer BN buffers  max|d|={dbn:.3e}")
print(f"  chosen glimpses    total flips={sum(flips)}/{sum(len(a) for a in idxE)}"
      f"   first differing step={first}")
print(f"  CONTROL no-eval vs no-eval (same code, repeat run): relL2={ctl:.3e}  "
      f"flips={ctl_flips}")
if dw == 0.0 and dbn == 0.0 and sum(flips) == 0:
    print("\nVERDICT: the periodic eval has NO side effect on training whatsoever.")
elif rel <= max(ctl * 3, 1e-6):
    print("\nVERDICT: the with-eval difference is WITHIN the run-to-run nondeterminism floor "
          "measured by the control -> no eval side effect beyond GPU nondeterminism.")
else:
    print("\nVERDICT: with-eval differs by MORE than the control floor -> a genuine eval "
          "side effect. Identify the channel (RNG vs state) before concluding it is a bias.")
