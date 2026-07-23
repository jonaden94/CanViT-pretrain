# 2026-06-14 — audit + clean full-train flow: COMPACTION HANDOFF

Honest state after a skeptical re-audit of the prior 48h. Read this top-to-bottom
before acting; several earlier numbers were wrong and are corrected here.

## VERIFIED ANCHORS (from runs/*/summary.json, c32, paper "scene" protocol)
- EG-C2F t1 = **41.04** (`egc2f_t2_c32`), t0 = **38.53**.
- VAL best-of-K oracle (R-IID, true CE) t1 = **44.05** (`valcand_seqoracle_t5_c32`); t0 38.53. = +3.0pp headroom at t1.
- TRAIN oracle t1 = **49.80** (`traincand_seqoracle_t5_c32`); t0 43.83.

## WHAT IS RUNNING (survives compaction; nohup on crockett ~/projects/CanViT-PyTorch-RL)
`rwr_fulltrain_flow_k64` — the CLEAN recipe the user asked for: single conditional
flow, NO critic, NO aux, NO subset. FULL ade20k train -> FULL val.
  cmd: `uv run python -m canvit_pytorch_rl.rwr_train --run-name rwr_fulltrain_flow_k64
        --tau 0.1 --k 64 --weighting soft --steps 60000 --eval-every 2000 --keep-every 10000`
  throughput ~1.0 step/s (k64 = 64 candidate forwards/step, ~4x k16); 60k steps ~16h;
  2526 steps/epoch (~42 min/ep). keep_every 10000 -> checkpoints exist for probing.
  Read progress: runs/rwr_fulltrain_flow_k64/metrics.jsonl, key `val_miou_t1_mode`.
  Background watcher bvngy9p34 was dumping the val_mode slope through ~ep3 (DIES at
  compaction — re-read metrics.jsonl directly instead).

## THE RECIPE, EXACTLY (files to read)
One-STEP t1 policy: act once from the full-scene t0 state.
- `rwr_train.py`: config (RWRConfig), train loop ~249-310, `riid_candidates` ~111,
  `candidate_canvas` ~149. Per step: t0 state (`full_scene_state`); conditioner sees
  canvas spatial tokens + probe ENTROPY (state only); propose **K=64 RANDOM (R-IID)**
  viewpoints (NOT flow samples); reward = frozen-probe per-image **CE** of each
  candidate's post-glimpse canvas; weights = softmax(z-scored(-CE)/tau=0.1); loss =
  -(w * logpi).sum.mean() = reward-weighted regression of the flow onto the low-CE
  candidates. Flow & conditioner grads clipped SEPARATELY to 1.0. AdamW, cosine, 5% warmup.
- `actor.py` CanvasActor: location-preserving conv conditioner (NEVER pooled) ->
  256-d context; deploy = `base_mode` (noise-0 flow action).
- `flow_head.py` SafeBoxFlow: 6-layer affine MAF over (cy,cx,scale) via atanh/logit
  safe-box bijection; context_init_scale 0 (flow ignores image at init).
- reward/eval: `rollout_eval.py` (`candidate_ce`, `full_scene_state`, `mle_weights`,
  `evaluate_rollout`); `candidate_data.z_targets`; `scoring.py`; `oracle.per_image_ce`.
In one line: amortize random-search-for-the-best-single-glimpse into a conditional
density; deploy the mode.

## AUDIT FINDINGS (skeptical re-derivation; corrects earlier claims)
SOLID:
- Anchors above. Eval-precision fix HOLDS: in-training `evaluate_rollout` t1 mode
  (41.1) == canonical `evaluate.py` base_mode (41.09); t0 both 38.53.
- self-density ranking is a BAD selector: actor_rank=advantage best-of-64 = **40.62**
  < mode (canonical base_mode = **41.09**). (`eval_cisaux_adv_k64`, `audit_cisaux_basemode`.)
WRONG / FRAGILE (do not repeat):
- "no-critic mode = 41.3" was a CHERRY-PICK (I read max over noisy evals via
  sort|tail). True settled/canonical mode = **41.1** = statistical TIE with EG-C2F.
- **The untrained (step-0) flow already scores val_mode = 41.1.** So trained mode ~=
  untrained mode ~= EG-C2F: a fixed image-independent glimpse gets ~41 at t1; training
  the MODE adds little on val. (BUT: rwr_fulltrain_flow_k64 went 41.1 -> 41.3 in 0.8 ep
  -- a real early move; judge by the SLOPE across many evals, not 2 points.)
- "flow+critic beats EG-C2F +0.46 (41.50)" is SEED-NOISE: same critic config seeds =
  41.22/41.26/41.32/41.55 (spread 0.33 ~= the margin). 41.5 is the lucky seed; mean
  ~41.34. The paired CI [+0.11,+0.53] was one seed, ignores seed variance.
SUBSET-ONLY (diagnostics, not policy results): all `rwr_ovf256_*` are trained on 256
  train imgs (1.3% of train). The plateau/mode-spread/coverage findings are on those.
  mode-spread ref 0.26 was computed as sorted(unique imgs)[:256] from candidates.parquet
  -- NEVER verified that ordering matches the trainer's Subset(range(256)). Possible confound.

## THE KEY OPEN QUESTION (the next decisive probe -- DO THIS FIRST after compaction)
Is the +3pp headroom a COVERAGE gap (the flow never samples the good viewpoint) or a
SELECTION gap (it samples it, nothing picks it)? Deploy=mode is a poor readout of a
multimodal flow, and self-density ranks badly -- but we have NOT measured whether the
flow's SAMPLES even cover the good region.
  Probe: `throwaway/flow_coverage_oracle.py <ckpt> <K>` -- best-of-K-by-true-CE over the
  flow's OWN samples on full val, vs a matched R-IID best-of-K (R-IID's ceiling = 44.05).
  (Mid-edit: I was adding the R-IID arm; finish that, then run on the first
  rwr_fulltrain_flow_k64 checkpoint -- runs/.../step_010000.pt -- NOT the subset ckpts.)
  INTERPRET: flow best-of-K ~44 -> coverage fine, SELECTION is the lever -> longrun =
  train a ranker/critic on the flow's proposals (the proven +path, but de-noise across
  seeds). flow best-of-K ~41-42 (<= R-IID) -> the flow proposes WORSE than random; the
  PROPOSER objective is broken -> longrun = fix it (sharper/again-on-policy candidates,
  different objective), a ranker won't help.

## EFFECTIVE LONGRUN -- decide via the coverage probe, do not just scale blindly
The current k64 run is a fine clean baseline but deploy=mode may cap it at ~the fixed-
glimpse floor. Before committing 16h: (1) run the coverage probe on its first
checkpoint; (2) pick proposer-fix vs ranker per the result; (3) whatever the longrun,
report on FULL val with TRAIN-scope labeled, and average >=3 seeds (the wins live inside
seed noise -- single-seed numbers are not evidence).

## DISCIPLINE (RULE TWO, CLAUDE.md, after this session's failures)
- Deliver the asked-for thing (clean flow, full train, full val); no aux/critic/subset
  unless explicitly chosen. Every mIoU states train-scope AND eval-scope; report in mIoU
  not nats; never headline a single noisy eval point or single seed.
- Killing: runs launched together get ADJACENT pids; map each pid->run via ppid+cmdline
  BEFORE killing (I killed the wrong run this session); verify the KEPT run still alive;
  never idle the GPU.

## EPHEMERAL (dies at compaction) -- re-establish from artifacts
Background watchers (bvngy9p34, b8nw2nlus, ...) and this reasoning. The nohup k64 run +
all runs/ artifacts + MLflow (crockett:5500, tunnel `ssh -L 5500:localhost:5500 crockett`)
survive. throwaways: flow_coverage_oracle.py (mid-edit), mode_vs_sample_spread.py.
