# 2026-07-08 — Q-Prop seeds, schema hardening, the comment audit (log)

Continues `2026-07-05-e2e-diff-single-image-probe.md` (which holds today's earlier entries through
the two-seed Q-Prop verdict). Chronological; entries appended as results land.

- pgqprop_s2 (seed 2, 640k, current main): first run with persisted `critic_state`; 1k eval 0.6951
  (family pace). Successor staged: pgsubz_s2 (third subtract-only-z seed) launches on its done line.
- **Three-seed Q-Prop verdict: 0.6861/0.6863/0.6861 (s0/s1/s2) = 0.6862±0.0001** — endpoint in-band
  (pgfloor 0.6865±0.0012) but with ~10x SMALLER seed spread and the consistent ~2x faster early
  convergence: the control variate buys variance reduction where theory predicts (consistency +
  speed), not a better optimum. pgqprop_s2's ckpts carry the critic (critic_state) for map analysis.
  pgsubz_s2 auto-chained and training.
- **pgsubz_s2: best 0.6859** — three subtract-only seeds now 0.6865/0.6860/0.6859 (0.6861±0.0003),
  INSIDE the pgfloor band and as tight as qprop's: the std division is confirmed droppable at band
  grade. pgsubz_s3 chained for the fourth seed; the ablation thread closes with it.

## Standing state at the 2026-07-08 decision point [written ~15:00 crockett]

- **Running**: nothing — GPU idle BY DECISION since pgsubz_s3 finished (15:56 crockett): the
  authorized backlog is empty and remaining fillers carry ~zero information; awaiting the user.
- **Threads closed today**: Q-Prop (3 seeds, 0.6862±0.0001 — in-band endpoint, ~2x faster, 10x
  tighter seeds; critic persisted from s2 on); subtract-only z (4 seeds 0.6858±0.0007, best-ever
  pg seed 0.6848); comment/docstring audit of the whole codebase; justfile recipe fix; mlflow renamed
  in place (canvit-pytorch-rl, id 5); session-log split.
- **Open decisions (user's)**: unfreezing ladder design (all prerequisites green: subtract-only z
  band-confirmed, entropy floor default, qprop available for tight bands); publish qprop/pgsubz
  artifacts?; new research direction vs idle.
- Monitoring: hourly cron (:23). Deploy discipline: push+reset one action (memory file).
- **pgsubz_s3: best 0.6848 — the strongest pg-family seed recorded, BELOW the qband mean.** The
  subtract-only-z thread closes at four seeds: 0.6865/0.6860/0.6859/0.6848 = **0.6858±0.0007** —
  slightly ahead of the pgfloor band (0.6865±0.0012) and statistically AT the qband (0.6853±0.0007).
  Dropping the std division is not merely free, it may be mildly beneficial (n=4, overlapping bands —
  "at least free" is the defensible claim). The unfreezing credit form is settled: subtract-only.
  **GPU now idle BY DECISION** (handoff block above): authorized backlog empty; unfreezing-ladder
  design, artifact publication, or a new direction await the user.
- Housekeeping: 138 intermediate step ckpts (2.9 GB) pruned from the retired sweep-trial dirs
  (dry-run reviewed; best.pt/last.pt/init/manifest/metrics kept everywhere — the permanent record).
- 16:29: GPU refilled with a genuine open question (cron tick + the repo's fill-with-queued-ablations
  rule): **pgqsub_s0/s1 = qprop x subtract-only combined — the EXACT unfreezing training config,
  never before run.** Open sub-question it answers: does the critic's MSE handle centered-but-unscaled
  targets (resid_std/critic_loss change scale) or does the combination need retuning before the
  ladder builds on it? References: qprop band 0.6862±0.0001, subz band 0.6858±0.0007.
- **pgqsub_s0: best 0.6855 — the strongest full-budget pg result of any configuration**, ON the qband
  mean (0.6853±0.0007) and ahead of both parents (qprop 0.6862±0.0001, subz 0.6858±0.0007). Critic
  diagnostics scale-consistent (residual/target fraction matches the standardized runs — the control
  variate is scale-invariant in practice). The unfreezing training config (qprop + subtract-only) is
  validated AND preferred on frozen perception. pgqsub_s1 chained (0.6931 @1k).
- **pgqsub_s1: 0.6845 — all-time best seed across every method incl. qreg.** Two-seed combined band
  0.6850±0.0007, nominally below / statistically tied with the qband (0.6853±0.0007). Seeds 2-3
  launched to make the claim band-grade: if n=4 holds this level, the pg family's best config
  MATCHES-OR-BEATS Q-regression outright — and it is exactly the unfreezing objective.
- **pgqsub_s2: 0.6841 — record again.** Three combined-config seeds: 0.6855/0.6845/0.6841 =
  **0.6847±0.0007, nominally AHEAD of the qband (0.6853±0.0007)**. s3 (4th seed) chains overnight;
  if it holds, tomorrow's headline: the actor's unfreezing objective BEATS Q-regression at matched
  budget — the first config to do so.
- **Four-seed combined band FINAL: 0.6855/0.6845/0.6841/0.6861 = 0.6851±0.0009 vs qband
  0.6853±0.0007 — statistically TIED with Q-regression, nominally ahead** (s3 pulled the trend back
  into overlap; "beats" is NOT the defensible claim, "matches with the best two seeds ever" is).
  The unfreezing objective (qprop + subtract-only) is validated, competitive, and the recommended
  pg config. GPU idle BY DECISION again — all bands complete, no authorized work; the unfreezing
  ladder, publication of pgqsub seeds, or new direction await the user.

## 2026-07-09 — THE UNFREEZING LADDER BEGINS (rung 1: probe)

- [user: "do whatever interesting thing you want"] Implemented `unfreeze=probe` (`4af8ba3`): one
  objective J = mean_t CE_t/CE_t0, two gradient routes, no loss weights — score-function credit for
  the policy (pgqsub machinery as validated), direct dJ/d(probe) computed IN the rollout per visited
  state (canvas detached -> backbone frozen; incremental backward; zero_grad moved pre-rollout; probe
  param group lr 1e-5 wd 0; probe_state persisted in ckpts; probe_grad_norm logged).
- **Smoke (200 steps): 0.7180 -> 0.6947 @100 -> 0.6905 @200** — frozen-perception methods need ~2k
  steps for that level; probe grads healthy (~1.9 pre-clip). The probe adapting to the glimpse-canvas
  distribution pays immediately, and the val eval uses the updated probe (genuine end-to-end val).
- **unfreeze_probe_s0 launched 10:42 crockett** (640k, seed 0; s1 chained). Judged against the pgqsub
  band 0.6851±0.0009 — the bet: adapting perception breaks the 0.685 frozen plateau. Caveats to watch:
  reward nonstationarity (the probe moves under the policy), probe overfit to train split (watch the
  train-slice fit vs val), and the t0 floor drifting (t0 CE now changes as the probe trains —
  cross-run t0 comparisons break for these runs).
- **[user skepticism, sustained by the data]**: the rung-1 gain decomposes as a UNIFORM probe lift
  (t0 floor −0.009) with the policy-conditional gap UNCHANGED (0.0798 -> 0.0782) — no co-adaptation
  signal. Chained s1 swapped for the discriminating CONTROL: `unfreeze_ctrl_s0` (policy frozen via
  lr 0, uniform random glimpses, probe trains identically) — if its val_ce_t0 drops the same ~0.009,
  the gain is generic probe fine-tuning and rung 1 is scaffolding, not a finding; the interesting
  question then lives at the backbone rung.
- **unfreeze_probe_s0 FINAL: best 0.6741** (frozen best band: 0.6851±0.0009). Decomposition holds at
  full budget: t0 floor −0.0102 (0.7649 -> 0.7547), policy-conditional gap +0.0008 (noise) — the gain
  is a uniform probe lift end to end. Control `unfreeze_ctrl_s0` running (relaunched 12:30: the staged
  command used `objective:qreg`; tyro kebab-cases the subcommand to `objective:q-reg` — parse-test
  staged commands, ~10 min idle). Verdict rule: control val_ce_t0 drops ~equally -> rung-1 gain is
  generic probe fine-tuning (scaffolding, not finding).

## End-of-session handoff [2026-07-09 ~13:35 crockett]

- **RUNNING**: `unfreeze_ctrl_s0` (probe fine-tune, frozen random policy, lr 0 / probe_lr 1e-5;
  /tmp/unfreeze_ctrl_s0.log; done ~14:00 crockett). NOTHING staged behind it — after it, GPU idles.
  Harvest: `grep -oE "best objective [0-9.]+" /tmp/unfreeze_ctrl_s0.log` and compare its val_ce_t0
  trajectory to the treatment's. **Verdict already effectively in at 6k: control t0 0.7542 ==
  treatment t0 0.7545 — the rung-1 probe lift is GENERIC fine-tuning, not an active-vision effect
  [user's skepticism vindicated].** Classification: rung 1 = validated scaffolding; its machinery
  (unfreeze flag, direct dJ/d(probe) in-rollout, probe_state ckpts) is ready for the backbone rung.
- **Best system result**: unfreeze_probe_s0 best 0.6741 (a uniform probe lift over the frozen
  0.6851±0.0009 plateau). Best frozen bands: pgqsub 0.6851±0.0009 (tied w/ qband 0.6853±0.0007).
- **Open decisions**: backbone rung (the real co-adaptation question) vs park unfreezing; publish
  pgqsub/unfreeze artifacts; README results refresh. crockett checkout at latest main; both remotes
  synced; `uv run just` green both environments; hourly session cron dies with this session.
