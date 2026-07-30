"""MULTI-STEP identity: do the two trainers stay identical over many optimizer steps?

`diff_training_trace.py` only proves ONE step from a fresh scorer. That cannot see anything
that ACCUMULATES — evolving BatchNorm running stats, the per-depth reward standardizers, the
eps-greedy RNG stream, or a weight trajectory that drifts. Ten harness runs sitting below
five `rl_train` runs on mIoU t4 (p=0.078) is not obviously noise, so the identity claim has
to be tested where it can actually fail.

Method, per step, on ONE canonical trajectory (so a divergence at step k is attributable to
step k rather than to accumulated drift):

  1. snapshot the scorer (weights + BN buffers) and both reward standardizers
  2. restore, re-seed, run `rl_train.rollout_and_loss` -> gradient A, mutated BN/norm state A
  3. restore, re-seed identically, run `harness.run_rollout`  -> gradient B, state B
  4. compare A vs B: the GRADIENT, the BN running stats, and the standardizer state
  5. advance the canonical trajectory with A's gradient through a real AdamW+LambdaLR

Both paths get the SAME batch in the same order and generators re-seeded to the same value
each step, so the eps-greedy draws line up. prime_on_policy is the real recipe's 0.5, so the
eps-greedy path is exercised rather than bypassed.

Usage: python diff_training_multistep.py [--steps 20] [--batch-size 16]
"""
import argparse
import copy

import torch

from canvit_pretrain.ade20k.config import Ade20kConfig
from canvit_pretrain.ade20k.data import make_ade20k_loaders
from canvit_pretrain.ade20k.rl_train import PolicyTrainConfig, build_action_table, rollout_and_loss
from canvit_pretrain.harness.loop import apply_requires_grad
from canvit_pretrain.harness.rollout import run_rollout
from canvit_pretrain.harness.spec import TrainSpec, fixed_horizon_bptt
from canvit_pretrain.tasks.ade20k.task import POLICY_FEATURE_GROUPS, Ade20kRunTask
from canvit_pretrain.train.config import JointPolicyConfig
from canvit_pretrain.train.rl import RunningNorm

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--prime", type=float, default=0.5, help="0.5 = the real recipe (eps-greedy on)")
ap.add_argument("--self-check", action="store_true",
                help="run rl_train as BOTH paths. Must print 0.000e+00 everywhere; if it does "
                     "not, this script is broken and its A-vs-B verdict means nothing. Two "
                     "earlier diff scripts of mine reported fake divergences for exactly this "
                     "reason, so never trust an A-vs-B run without seeing this pass first.")
args = ap.parse_args()

T, dev = 5, torch.device("cuda")
HORIZON = T - 1
torch.manual_seed(0)

cfg = Ade20kConfig(
    resize_mode="squish", scene_size=512, canvas_grid=64, augment=False, mode="frozen",
    probe_repo="canvit/probe-ade20k-40k-s512-c64-in21k", n_timesteps=T,
    eval_policy="policy", batch_size=args.batch_size, num_workers=2)
task = Ade20kRunTask(cfg)
task.rl = JointPolicyConfig(use_rl=True, feature_groups=POLICY_FEATURE_GROUPS,
                            prime_on_policy=args.prime, select_bn_eval=True)

model, head = task.build_model(dev, prior_model_config=None)
cg = task.canvas_grid(model)
# The horizon MUST come from cfg.n_timesteps, exactly as `cli.resolve_spec` does it.
# `TrainSpec.policy_only()` defaults to BpttSpec(horizon=10); taking that default silently
# compares rl_train at horizon 4 against the harness at horizon 9, which "diverges" for
# entirely uninteresting reasons. (It did, on the first run of this script.)
spec = TrainSpec.policy_only(freeze_model=True,
                             bptt=fixed_horizon_bptt(frozen=True, horizon=T))
assert spec.bptt.horizon == T, spec.bptt
joint = task.build_policy(model, device=dev, canvas_grid=cg,
                          generator=torch.Generator(device=dev).manual_seed(0))
apply_requires_grad(model=model, head=head, joint=joint, spec=spec)
model.eval()
joint.scorer.train()
scorer = joint.scorer

rl_cfg = PolicyTrainConfig(canvas_grid=cg, score_res=128, train_horizon=HORIZON,
                           batch_size=args.batch_size, select_bn_eval=True,
                           resize_mode="squish", scene_size=512, prime_on_policy=args.prime)
obj = rl_cfg.build_objective()
vp_flat, _ = build_action_table(model, rl_cfg)
vp_flat = vp_flat.to(dev)

# rl_train's optimizer recipe, driving the ONE canonical trajectory.
opt = torch.optim.AdamW(list(scorer.parameters()), lr=rl_cfg.lr,
                        weight_decay=rl_cfg.weight_decay,
                        betas=(rl_cfg.adam_beta1, rl_cfg.adam_beta2))
warm = max(1, int(rl_cfg.warmup_frac * rl_cfg.max_steps))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))

amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)

# Same batches, same order, for both paths.
train_loader, _ = make_ade20k_loaders(cfg)
it = iter(train_loader)
batches = [next(it) for _ in range(args.steps)]

runningA = [RunningNorm(momentum=rl_cfg.target_momentum, device=dev) for _ in range(HORIZON)]


_bn_calls = {"n": 0}


def _bn_hook(mod, _inp):
    if mod.training:
        _bn_calls["n"] += 1


for _m in scorer.modules():
    if isinstance(_m, torch.nn.BatchNorm2d):
        _m.register_forward_pre_hook(_bn_hook)


def grads() -> torch.Tensor:
    return torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel(), device=dev)
                      for _, p in sorted(scorer.named_parameters())])


def bn_state() -> torch.Tensor:
    return torch.cat([b.flatten().float() for n, b in sorted(scorer.named_buffers())
                      if "running" in n or "num_batches" in n])


def norm_vec(objs) -> torch.Tensor:
    return torch.stack([torch.stack([o.mean, o.sq, torch.tensor(float(o.count), device=dev)])
                        for o in objs]).flatten()


def set_norms(objs, vec: torch.Tensor) -> None:
    v = vec.reshape(-1, 3)
    for o, row in zip(objs, v):
        o.mean, o.sq, o.count = row[0].clone(), row[1].clone(), int(row[2].item())


print(f"steps={args.steps} batch={args.batch_size} prime_on_policy={args.prime} "
      f"canvas_grid={cg}\n")
print(f"{'step':>4} {'|gradA|':>10} {'|gradB|':>10} {'max|dgrad|':>11} {'relL2':>9} "
      f"{'max|dBN|':>9} {'max|dnorm|':>10}  verdict")

worst = 0.0
for step in range(args.steps):
    images, masks = batches[step]
    images, masks = images.to(dev), masks.to(dev)

    snap_w = copy.deepcopy(scorer.state_dict())
    snap_n = norm_vec(runningA).clone()

    # --- path A: rl_train ---------------------------------------------------
    scorer.load_state_dict(snap_w)
    set_norms(runningA, snap_n)
    opt.zero_grad(set_to_none=True)
    _bn_calls["n"] = 0
    genA = torch.Generator(device=dev).manual_seed(1000 + step)
    lossA, _ = rollout_and_loss(
        seg=model, net=scorer, critic=None, encoder=joint.policy_selector.encoder,
        images=images, masks=masks, vp_flat=vp_flat, cfg=rl_cfg, obj=obj,
        running=runningA, log_alpha=None, gen=genA, amp_ctx=amp_ctx)
    lossA.backward()
    gA, bnA, nA = grads().clone(), bn_state().clone(), norm_vec(runningA).clone()
    bn_calls_A = _bn_calls["n"]

    # --- path B: harness (or rl_train again, under --self-check) -------------
    scorer.load_state_dict(snap_w)
    opt.zero_grad(set_to_none=True)
    _bn_calls["n"] = 0
    if args.self_check:
        runningB = [RunningNorm(momentum=rl_cfg.target_momentum, device=dev) for _ in range(HORIZON)]
        set_norms(runningB, snap_n)
        lossB, _ = rollout_and_loss(
            seg=model, net=scorer, critic=None, encoder=joint.policy_selector.encoder,
            images=images, masks=masks, vp_flat=vp_flat, cfg=rl_cfg, obj=obj,
            running=runningB, log_alpha=None,
            gen=torch.Generator(device=dev).manual_seed(1000 + step), amp_ctx=amp_ctx)
        lossB.backward()
    else:
        joint.running.clear()
        runningB = [joint._norm(d) for d in range(1, HORIZON + 1)]  # create in depth order
        set_norms(runningB, snap_n)
        joint.policy_selector.generator = torch.Generator(device=dev).manual_seed(1000 + step)
        bound = task.bind((images.cpu(), masks.cpu()), dev, model=model, head=head)
        run_rollout(model=model, images=images, task=bound, selector=joint.random_selector,
                    bptt=spec.bptt, branches=task.branches(), canvas_grid_size=cg,
                    amp_ctx=amp_ctx, task_weight=spec.task_weight, joint=joint)
    gB, bnB, nB = grads().clone(), bn_state().clone(), norm_vec(runningB).clone()
    # Structural check BEFORE any numeric comparison: if the two paths did not even run the
    # scorer the same number of times, the gradients are not comparable and a numeric verdict
    # would be meaningless. This is what caught the horizon-10 bug.
    assert bn_calls_A == _bn_calls["n"], (
        f"step {step}: train-mode scorer forwards differ — rl_train {bn_calls_A} vs "
        f"harness {_bn_calls['n']}. The two paths are not running the same rollout shape.")

    dg = (gA - gB).abs().max().item()
    rel = ((gA - gB).norm() / gA.norm().clamp_min(1e-12)).item()
    dbn = (bnA - bnB).abs().max().item()
    dn = (nA - nB).abs().max().item()
    worst = max(worst, rel)
    # 5e-5: the --self-check floor over 20 steps measured 6.3e-6, and A-vs-B legitimately
    # differs in fp REDUCTION ORDER (harness: per-depth mse summed; rl_train: one mse over the
    # concatenated depths) — mathematically identical, ~1e-5 relative on GPU. BN buffers and
    # the standardizers have no such excuse and are held to bit-identity.
    ok = "ok" if rel < 5e-5 and dbn == 0.0 and dn == 0.0 else "*** DIVERGED ***"
    print(f"{step:>4} {gA.norm().item():>10.4f} {gB.norm().item():>10.4f} {dg:>11.3e} "
          f"{rel:>9.2e} {dbn:>9.2e} {dn:>10.2e}  {ok}")

    # --- advance the ONE canonical trajectory with path A -------------------
    scorer.load_state_dict(snap_w)
    set_norms(runningA, nA)
    opt.zero_grad(set_to_none=True)
    for p, g in zip((p for _, p in sorted(scorer.named_parameters())), gA.split(
            [p.numel() for _, p in sorted(scorer.named_parameters())])):
        p.grad = g.view_as(p).clone()
    torch.nn.utils.clip_grad_norm_(scorer.parameters(), rl_cfg.grad_clip)
    opt.step()
    sched.step()
    # carry path A's BN buffers forward (load_state_dict above restored the pre-step ones)
    with torch.no_grad():
        i = 0
        for n, b in sorted(scorer.named_buffers()):
            if "running" in n or "num_batches" in n:
                k = b.numel()
                b.copy_(bnA[i:i + k].view_as(b).to(b.dtype))
                i += k

print(f"\nworst relative gradient difference over {args.steps} steps: {worst:.3e}")
print("VERDICT:", "IDENTICAL across all steps (within the fp reduction-order floor)"
      if worst < 5e-5 else "DIVERGENT")
