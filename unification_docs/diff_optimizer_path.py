"""Given the SAME gradient, do the two update paths produce the same weights?

`diff_training_multistep.py` drove ONE canonical optimizer (rl_train's) for both paths, so it
did not test the harness's own update machinery at all: `harness/optim.py::
build_optimizer_and_scheduler` (AdamW built from `spec.optim` + a per-group LambdaLR) and
`harness/loop.py`'s clip-then-step. rl_train instead builds AdamW inline and clips `params`.

This feeds an identical synthetic gradient to both and compares the resulting weights over
several steps, so Adam moment state and the LR ramp are exercised, not just step 1.
"""
import copy

import torch
from canvit_pytorch.policy import ViewpointScorer

from canvit_pretrain.ade20k.rl_train import PolicyTrainConfig
from canvit_pretrain.harness.optim import build_optimizer_and_scheduler
from canvit_pretrain.harness.spec import GroupOptim, ScheduleSpec, TrainSpec, fixed_horizon_bptt

STEPS, dev = 12, torch.device("cuda")
rl = PolicyTrainConfig()
warm = max(1, int(rl.warmup_frac * rl.max_steps))
print(f"lr={rl.lr} wd={rl.weight_decay} betas=({rl.adam_beta1},{rl.adam_beta2}) "
      f"clip={rl.grad_clip} warmup={warm}/{rl.max_steps}  steps={STEPS}")

torch.manual_seed(0)
net = ViewpointScorer(canvas_dim=1024, width=128, block_layers=3, n_scale=2,
                      centers_per_axis=16, scales=(0.5, 0.25), dueling=True).to(dev)
init = copy.deepcopy(net.state_dict())
params = [p for _, p in sorted(net.named_parameters())]

# Fixed synthetic gradients, one per step — identical for both paths.
g = torch.Generator(device=dev).manual_seed(7)
grads = [[torch.randn(p.shape, generator=g, device=dev) * 0.05 for p in params]
         for _ in range(STEPS)]


def run_rl_train():
    net.load_state_dict(init)
    opt = torch.optim.AdamW(list(net.parameters()), lr=rl.lr, weight_decay=rl.weight_decay,
                            betas=(rl.adam_beta1, rl.adam_beta2))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    traj = []
    for s in range(STEPS):
        opt.zero_grad(set_to_none=True)
        for p, gr in zip(params, grads[s]):
            p.grad = gr.clone()
        torch.nn.utils.clip_grad_norm_(net.parameters(), rl.grad_clip)
        opt.step(); sched.step()
        traj.append((sched.get_last_lr()[0], torch.cat([p.detach().flatten() for p in params]).clone()))
    return traj


def run_harness():
    net.load_state_dict(init)
    # Exactly what cli.resolve_spec builds for the policy group, and what loop.py then does.
    spec = TrainSpec.policy_only(freeze_model=True, bptt=fixed_horizon_bptt(frozen=True, horizon=5),
                                 optim={"policy": GroupOptim(
                                     lr=rl.lr, weight_decay=rl.weight_decay,
                                     betas=(rl.adam_beta1, rl.adam_beta2),
                                     schedule=ScheduleSpec(kind="warmup_constant", warmup_steps=warm))})
    opt, sched = build_optimizer_and_scheduler(spec, {"policy": list(net.parameters())})
    scorer_ids = {id(p) for p in net.parameters()}
    trainable = [p for grp in opt.param_groups for p in grp["params"] if id(p) not in scorer_ids]
    traj = []
    for s in range(STEPS):
        opt.zero_grad(set_to_none=True)
        for p, gr in zip(params, grads[s]):
            p.grad = gr.clone()
        if trainable:                                     # empty for policy_only
            torch.nn.utils.clip_grad_norm_(trainable, rl.grad_clip)
        torch.nn.utils.clip_grad_norm_(net.parameters(), rl.grad_clip)   # loop.py's scorer clip
        opt.step(); sched.step()
        traj.append((sched.get_last_lr()[0], torch.cat([p.detach().flatten() for p in params]).clone()))
    return traj


A, B = run_rl_train(), run_harness()
print(f"\n{'step':>4} {'lr rl_train':>13} {'lr harness':>12} {'max|dW|':>11} {'relL2':>10}")
worst = 0.0
for s, ((la, wa), (lb, wb)) in enumerate(zip(A, B)):
    d = (wa - wb).abs().max().item()
    rel = ((wa - wb).norm() / wa.norm()).item()
    worst = max(worst, rel)
    print(f"{s:>4} {la:>13.3e} {lb:>12.3e} {d:>11.3e} {rel:>10.3e}")
print(f"\nworst relative weight difference: {worst:.3e}")
print("VERDICT:", "IDENTICAL" if worst == 0.0 else ("within fp noise" if worst < 1e-6 else "DIVERGENT"))
