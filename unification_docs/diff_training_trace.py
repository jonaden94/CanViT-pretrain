"""Where exactly do the two training rollouts diverge? Trace both, depth by depth.

`diff_training_step.py` showed the scorer gradients differ even at prime_on_policy=1.0 (no
eps-greedy RNG), and that `reward_frac` differs by ~80% — far too much for a rounding effect.
This walks each path with ITS OWN primitives and prints, per depth, the reward-CE and the
chosen candidate, so the first point of divergence is visible rather than inferred.

Deterministic by construction: prime_on_policy=1.0 => pure argmax, no RNG anywhere.
Both paths share ONE scorer and ONE encoder, with the scorer's BatchNorm restored between
runs (a train-mode rollout mutates it).
"""
import copy

import torch
from canvit_pytorch.policy import head_logits

from canvit_pretrain.ade20k.config import Ade20kConfig
from canvit_pretrain.ade20k.data import make_ade20k_loaders
from canvit_pretrain.ade20k.rl_train import (
    PolicyTrainConfig,
    advance_state,
    build_action_table,
    ce_from_logits,
    full_scene_state,
)
from canvit_pretrain.harness.loop import apply_requires_grad
from canvit_pretrain.harness.spec import TrainSpec
from canvit_pretrain.tasks.ade20k.task import POLICY_FEATURE_GROUPS, Ade20kRunTask
from canvit_pretrain.train.config import JointPolicyConfig
from canvit_pretrain.train.viewpoint import ViewpointType

T, dev = 5, torch.device("cuda")
HORIZON = T - 1
torch.manual_seed(0)

cfg = Ade20kConfig(
    resize_mode="squish", scene_size=512, canvas_grid=64, augment=False, mode="frozen",
    probe_repo="canvit/probe-ade20k-40k-s512-c64-in21k", n_timesteps=T,
    eval_policy="policy", batch_size=16, num_workers=2)
task = Ade20kRunTask(cfg)
task.rl = JointPolicyConfig(use_rl=True, feature_groups=POLICY_FEATURE_GROUPS,
                            prime_on_policy=1.0, select_bn_eval=True)

model, head = task.build_model(dev, prior_model_config=None)
cg = task.canvas_grid(model)
spec = TrainSpec.policy_only(freeze_model=True)
joint = task.build_policy(model, device=dev, canvas_grid=cg,
                          generator=torch.Generator(device=dev).manual_seed(0))
apply_requires_grad(model=model, head=head, joint=joint, spec=spec)
model.eval()
joint.scorer.train()

train_loader, _ = make_ade20k_loaders(cfg)
images, masks = next(iter(train_loader))
images, masks = images.to(dev), masks.to(dev)
B = images.shape[0]

snapshot = copy.deepcopy(joint.scorer.state_dict())
amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
net, encoder = joint.scorer, joint.policy_selector.encoder

rl_cfg = PolicyTrainConfig(canvas_grid=cg, score_res=128, train_horizon=HORIZON,
                           batch_size=B, select_bn_eval=True, resize_mode="squish", scene_size=512)
vp_flat, _ = build_action_table(model, rl_cfg)
vp_flat = vp_flat.to(dev)


def restore():
    joint.scorer.load_state_dict(snapshot)
    joint.scorer.zero_grad(set_to_none=True)
    joint.running.clear()


def select(feats):
    """Mode (b) selection EXACTLY as `rollout_and_loss` does it.

    The train-mode forward comes FIRST and is not optional: it is the one that carries the
    loss, and it UPDATES `frontend.bn`'s running statistics, which the eval-mode selection
    forward then reads. Omitting it (as an earlier version of this script did) leaves path A
    selecting on stale BN stats while path B selects on updated ones — which manufactures a
    divergence at t1 that is entirely an artifact of the measuring script.
    """
    net(feats.float())                      # train-mode: updates frontend.bn running stats
    was = net.training
    net.eval()
    with torch.no_grad():
        s = net(feats.float()).reshape(B, -1)
    if was:
        net.train()
    return s.argmax(dim=1)


# --- path A: rl_train primitives -------------------------------------------------
restore()
encoder.reset()
with torch.no_grad(), amp_ctx:
    stA = full_scene_state(model, images, canvas_grid=cg, glimpse_px=rl_cfg.glimpse_px)
    logitsA = head_logits(model, stA.canvas, canvas_grid=cg)
    ceA = ce_from_logits(logitsA, masks, score_res=rl_cfg.score_res)
traceA = [("t0", ceA.mean().item(), None)]
for t in range(HORIZON):
    with torch.no_grad():
        f = encoder(stA, logits=logitsA).float()
    idx = select(f)
    with torch.no_grad(), amp_ctx:
        stA = advance_state(model, images, stA, vp_flat[idx], rl_cfg.glimpse_px)
        logitsA = head_logits(model, stA.canvas, canvas_grid=cg)
        ceA = ce_from_logits(logitsA, masks, score_res=rl_cfg.score_res)
    traceA.append((f"t{t + 1}", ceA.mean().item(), idx.clone()))

# --- path B: harness primitives --------------------------------------------------
restore()
bound = task.bind((images.cpu(), masks.cpu()), dev, model=model, head=head)
sel = joint.policy_selector
ctx = sel.start_rollout(t0_type=ViewpointType.FULL, batch_size=B, device=dev)
with amp_ctx:
    vp0 = sel.select(vp_type=ViewpointType.FULL, ctx=ctx, t=0, batch_size=B, device=dev,
                     state=model.init_state(batch_size=B, canvas_grid_size=cg))
    from canvit_pretrain.harness.rollout import _to_vp
    goutB = bound.forward_glimpse(model=model, images=images,
                                  state=model.init_state(batch_size=B, canvas_grid_size=cg),
                                  viewpoint=_to_vp(vp0), backbone_no_grad=True)
ceB = bound.per_image_loss(goutB.readout)
traceB = [("t0", ceB.mean().item(), None)]
stateB = goutB.state
for t in range(1, T):
    vpn = sel.select(vp_type=ViewpointType.RANDOM, ctx=ctx, t=t, batch_size=B, device=dev,
                     state=stateB)
    idxB = sel.last_aux["flat_idx"]
    with amp_ctx:
        goutB = bound.forward_glimpse(model=model, images=images, state=stateB,
                                      viewpoint=_to_vp(vpn), backbone_no_grad=True)
    ceB = bound.per_image_loss(goutB.readout)
    stateB = goutB.state
    traceB.append((f"t{t}", ceB.mean().item(), idxB.clone()))

# --- report ----------------------------------------------------------------------
print(f"\n{'depth':6s} {'CE rl_train':>12s} {'CE harness':>12s} {'dCE':>10s}   idx agree")
for (na, ca, ia), (nb, cb, ib) in zip(traceA, traceB):
    agree = "" if ia is None else f"{int((ia == ib).sum())}/{B}"
    print(f"{na:6s} {ca:12.6f} {cb:12.6f} {cb - ca:+10.6f}   {agree}")

print("\nreward_frac (mean over depths of (prev-cur)/prev):")
for nm, tr in (("rl_train", traceA), ("harness ", traceB)):
    ces = [c for _, c, _ in tr]
    fr = [(ces[i] - ces[i + 1]) / ces[i] for i in range(len(ces) - 1)]
    print(f"  {nm}  {sum(fr) / len(fr):+.6f}   per-depth {[round(x, 5) for x in fr]}")
