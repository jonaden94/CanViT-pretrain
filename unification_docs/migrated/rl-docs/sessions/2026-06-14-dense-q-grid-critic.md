# 2026-06-14 — dense Q-grid critic (U-Net over the viewpoint pyramid): investigation + handoff

Single entry point for picking this thread up cold. Builds on the reward-landscape
audit (CE↔IoU misalignment) earlier the same day; see memory `reward-ce-iou-misalignment`
and `always-log-val-ce-with-miou`.

## THE IDEA (validated direction, not yet a result)
Replace the POINTWISE critic (boxent: one action -> one scalar, K forwards to rank K)
with a DENSE one: canvas features -> the WHOLE value grid over the action space in one
forward. A dense Q-map over discretized viewpoints IS the policy (argmax = greedy
deploy; no separate generator). Semantic-segmentation-shaped: input canvas spatial
feats [B,1024,32,32] (+ entropy), output a per-viewpoint value grid. Q-learning framing.

## WHAT'S VALIDATED (cheap, this session)
- **Pyramid/U-Net bias is real.** On the canvas's natural pyramid (level R: RxR center
  grid tiling, scale s=1/R), the aligned cross-scale test (coarse cell vs mean of its
  2x2 fine children) gives r=0.44..0.76 — coarse ~ pooled fine + residual. So shared
  encoder + skip-decoder + per-level heads is justified. ("where" matters at every
  scale; spatial std 0.6-5.5pp.) [pyramid12.pt, n=12 per-image IoU — directional.]
  NOTE: my earlier "scales independent (r=0.03)" was a MISALIGNMENT ARTIFACT (compared
  scales at a fixed center grid). Retracted.
- **Decouple stride from scale -> use OVERLAPPING glimpses.** Tiling forces stride=size
  (coarse scales collapse to 2x2/1x1 centers — no placement resolution). A FIXED CxC
  center grid at every scale (overlap) keeps placement res everywhere. Concrete win:
  overlap {0.5,0.25}@16x16 oracle ceiling = +7.79pp vs tiling-6-scales +6.42pp on the
  same val imgs — **placement resolution buys more than scale coverage.**
- **Arch is expressive + trains fast.** QUNet (tiling AND overlap variants) overfits the
  12-img dense targets perfectly (train corr 1.0 per level, loss<1e-3 by ~step 500) and
  reproduces train maps faithfully (qunet_overlap_train.png). Generalization UNtested:
  val corr ~0 at n=8 — purely data-limited, NOT capacity. Both archs equal here.
- **Where the headroom lives (scale).** Full-val CE-oracle (scale_restriction_oracle.py,
  valcand_rollout_egc2f t1, unrestricted=44.05 ✓): cumulative s<0.55 -> 43.50, s<0.70 ->
  43.88; s>=0.85 alone -> 40.73 (≈ t0/EG-C2F floor). **Big glimpses (>0.7) add ~nothing.**
  Per-image pyramid: s=0.25 best single (+4.87pp, ~78% of +6.22 achievable), {0.5,0.25}
  ~+5.41pp (~87%), sub-0.25 zoom adds ~+0.8pp more (~13%) that the flow's scale_min=0.25
  currently FORECLOSES — a real lever (relax scale_min).

## SCOPE LOCKED (for the next real run)
- Scales **{0.5, 0.25}** (two aligned, deployable, ~87% of achievable). 0.25-only (~78%,
  single 2D heatmap, no scale axis) is the simpler fallback. Sub-0.25 is a later lever.
- **16x16 overlapping** center grid (decoupled stride). Target = **z-scored gain-over-t0**
  (per-image z over all scales/cells: removes the 6%->77% offset, keeps cross-level
  comparability, forces pattern-fitting over mean-regression).
- Arch = OverlapUNet (qunet_overlap.py): encoder 32->1 (global bottleneck), decode to
  16x16 w/ skips, per-scale 1x1 head -> [B,n_scale,16,16].

## THE NEXT STEP (first real compute commitment)
Collect dense {0.5,0.25} x 16x16 overlap targets on a REAL train/val split (hundreds of
scenes) -> train OverlapUNet -> read held-out corr + deploy regret (does the bias
generalize?). Cost 512 glimpses/scene; e.g. 512 train + 128 val ≈ 328k glimpses ≈ ~35
min CONTENDED on crockett (faster solo). Scale up if 512 underfits the mapping. Decision
pending from user: image count + run alongside k64 vs hold.

## TOOLING (all committed; throwaway/)
PIPELINE (keepers): dump_pool_features.py (canvas feats for local nets) | reward_overlap_collect.py
(fixed CxC overlap targets) | qunet_overlap.py (the dense-Q U-Net; VIZ=train|val) |
reward_map_analyze.py (figures). PROBES (one-off): reward_map_collect.py (common-grid
landscape) | reward_pyramid_collect.py + reward_pyramid_viz.py (natural pyramid + aligned
cross-scale) | reward_map_structure.py (smoothness/alignment) | scale_restriction_oracle.py
(achievable mIoU vs scale band) | critic_map_collect.py (boxent critic over grid) |
dump_pool_images.py (RGB scenes).
Run pattern: edit local -> commit -> `git push deploy main` -> on crockett `git fetch -q
origin && git checkout -q <rev>` -> run. Local analysis: `uv run --no-project python ...`.
ALWAYS show scenes beside heatmaps [user, hard rule].

## ARTIFACTS (outputs/reward_maps/, gitignored; on crockett + pulled local)
pool12.pt (18x18x4 common-grid reward), pool12_rgb.pt, pool12_feats.pt [12,1024,32,32],
critic_pool12.pt, pyramid12.pt (tiling 1..32), overlap.pt ({0.5,0.25,0.125,0.0625}x16x16).
Pool idxs = evenly-spaced val [0,182,363,545,727,909,1090,1272,1454,1636,1817,1999].

## OPEN DESIGN QUESTIONS
- Per-scale head should tap encoder depth matching that scale's RF (FPN/atrous) vs the
  current shared-decoder heads — try once it generalizes.
- Oracle target by-IoU (true ceiling) vs by-CE (what we used in scale_restriction).
- Relax scale_min below 0.25 to capture the zoom headroom.

## CONCURRENT (survives): rwr_fulltrain_flow_k64
Clean flow baseline, commit cdf6647, `--tau 0.1 --k 64 --weighting soft --steps 60000`,
cis=0, scale_min 0.25. As of ~step 6000: val_mode flat ≈41.1 (= untrained constant-crop
floor; the 41.25 blip didn't hold), val_samp creeping 40.7->41.2, spreads y/x/scale all
rising from 0. Learning conditionality; deploy-mode not yet beating the floor (consistent
with CE-misalignment). Throughput slowed by this session's collection jobs sharing GPU.
train_mode = full deterministic pass over FIRST 1000 train imgs (fixed slice, not full
train, not a batch; mode is deterministic -> no eval noise).
