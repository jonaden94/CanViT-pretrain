# exp28 — pretraining replication with one x0.1 LR drop

Re-runs the four exp22 pretrains **from scratch on the current unified code base**, each with
a single x0.1 LR drop followed by 204,800 more steps. This is the final empirical check that
the unification did not change pretraining.

## Two phases, because there is no in-run LR-drop feature

The harness has `warmup_constant` / `warmup_cosine` / `warmup_onecycle` — none has a
"x0.1 at step N" milestone. exp22 achieved the drop the same way this does: a **separate run**
seeded from a checkpoint file, with a flat lower LR.

| phase | launcher | LR | length |
|---|---|---|---|
| A | `exp28-<arm>.sh` | warmup 100k -> constant 4e-4 | to the drop step |
| B | `exp28-<arm>-lrdrop.sh` | flat 4e-5, `warmup_steps=0` | 25 jobs = 204,800 steps |

**The drop point is a FILENAME, not a step comparison** (`CFG_SEED_CKPT=.../step-<N>.pt`), so
it cannot fire early, late, or twice no matter how many array tasks fail. Phase B's launcher
refuses to submit until that file exists.

| arm | phase-A jobs | drop step | phase B |
|---|---|---|---|
| `exp28-uniform16-teacherinit` | 77 | 630,784 | yes |
| `exp28-fovi-teacherinit` | 138 | 1,130,496 | yes |
| `exp28-uniform16` | 176 | 1,441,792 | yes |
| `exp28-fovi` | 245 (full 2,007,040) | none | no |

## The array is a BUDGET, not a schedule

`job_index` comes from the checkpoint's `resume_state`, **not** `SLURM_ARRAY_TASK_ID`
(`distill/task.py:449`), so the step count advances only for jobs that *succeed* — no shard is
skipped or re-read when a task dies. Consequence: if tasks fail, phase A ends **below** its
target step. Re-submit the remainder until `step-<N>.pt` exists, and only then launch phase B.
`resume_start_step` hard-errors rather than guessing if a checkpoint's scheduler step disagrees
with the schedule-derived step, so a mid-job save cannot silently shift anything.

## Accepted differences vs exp22 (owner-confirmed 2026-07-31)

1. **`normalizer_shards = 4`** (current default) vs exp22's effective 1 shard / all samples —
   exp22's pin `fe24aa1` had no such knob, only `normalizer_max_samples=0`. Pooling 4 shards is
   the more accurate estimate but shifts the target normalization, so **the loss scale is not
   identical and curves will not overlay exp22's exactly.**
2. **Entry point** `canvit_train.harness.run` vs the deleted `canvit_pretrain.train` — already
   A/B-verified in exp23/exp26; verifying it at full scale is the point of exp28.
3. **`PYTORCH_COMMIT 3277048 -> 1f5121b`** — verified irrelevant to distill: the diff touches
   only `data/ade20k.py`, `metrics.py`, `model/classification/`, `model/segmentation/` and
   `policy/`. Nothing in the distill training path (no `model/pretraining`, `backbone`,
   `patcher`, `attention`, `rope`, `standardizers`), so pretraining numerics are unchanged.
4. `exp22-fovi` and `exp22-fovi-teacherinit` originally suffered a **mid-array pin change**
   (`d2f7b50`) that jumped their eval scale by +0.058. exp28 has no such discontinuity, which
   is better but means the new fovi curve will not track the old one below its break step.

Everything else — `peak_lr 4e-4`, warmup 100k, batch 64, `steps_per_job 8192`,
`val_every 8192`, `log_every 512`, 4 workers, the full foveated patcher geometry, the
teacher-init flag — was copied value-for-value from the exp22 launchers.

## Cost

~1,420 GPU-h. At `%1` per array: `exp28-fovi` ~20 days, the three drop parents ~12/15/6 days.
Raise `%1` or split the arrays if that is too slow.

## Verification: exp22's step-8192 loss (the first comparable checkpoint)

Each exp22 parent's FIRST array job ended at step 8192 with these `train_loss` values, read
from that run's earliest job log (so unaffected by array ordering). exp28's step-8192 losses
should land near them; a systematic offset quantifies the 4-shard-vs-1-shard normalizer effect.

| arm | exp22 @ step 8192 (1 shard, `shard-001751`) |
|---|---|
| `uniform16` | 1.8945 |
| `uniform16-teacherinit` | 1.6106 |
| `fovi` | 1.8885 |
| `fovi-teacherinit` | 1.8377 |

Note the normalizer differs in **identity as well as count**: exp22 drew its single shard from
the seed-dependent shuffled order (`shard-001751`), while current code globs the sorted head
(`shard-000000..000003`) for a seed-independent pick.

## Submitted job IDs (2026-07-31)

| run | job |
|---|---|
| `exp28-uniform16-teacherinit` | 15113377 |
| `exp28-fovi-teacherinit` | 15113378 |
| `exp28-uniform16` | 15113379 |
| `exp28-fovi` | 15113380 |
