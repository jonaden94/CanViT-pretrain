# exp31 — ADE20K viewpoint policy (QReg), 10 seeds, harness, current code

Ten seeds of the harness-ported Q-regression viewpoint policy. Recipe = exp27's **`lossfix`**
arm verbatim; only the pins and run group change.

## Why `lossfix` and not `bneval`/`pooled`

exp27's arms split by ENTRY POINT, and the names do not advertise it:

| arm | submits | what it is |
|---|---|---|
| `oldloop`, `bneval`, `pooled` | `slurm/archive/ade20k/train_policy.sbatch` | the **deleted `canvit_pretrain.ade20k.rl_train`** — the reference |
| `harness`, **`lossfix`** | `slurm/harness_train.sbatch` | the **harness** — the ported path |

So the 10 existing `bneval` seeds are *reference* runs, not harness runs. `lossfix` is the
harness path at pin `bc0b16b`, i.e. after the one real harness/rl_train divergence was fixed
(the policy gradient was 0.8x); after that fix the two paths are bit-identical. That is "our
own ported q policy", which is what exp31 re-verifies. (`bneval`'s header is inherited text
from the old-loop arm and reads as if it were the reference — easy to get wrong.)

## The frozen backbone: no extraction needed

The published ade20k-policy HF checkpoints contain **no backbone**. `best.pt` holds only
`net_state` — 452 tensors, 5.68M params, 22.7MB safetensors — the `ViewpointScorer`. Policy
training freezes the backbone, so it is never saved.

It is also unnecessary: all 21 published policy checkpoints record
`model_repo = canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02`, and
that string is already `Ade20kConfig.model_repo`'s default (`ade20k/config.py:48`). So the
premise — one identical backbone shared by every published policy — is correct and verified;
it just needs no flag. `CFG_MODEL_REPO` is deliberately unset.

That backbone is the published Feb-2026 pretrain, **not** any exp22/exp28 model, so exp31 is
independent of exp28/29/30 and directly comparable to exp27.

## Submit

```bash
for s in 0 1 2 3 4 5 6 7 8 9; do SEED=$s bash slurm/runs/exp31_policy_qreg_10seed/policy-qreg-s0.sh; done
```

`CFG_RESIZE_MODE=squish` is the measurement contract the qband band is defined by. It has
silently regressed once (`1a0b452` defaulted it to `center_crop`; an arm landed 0.016 "better"
than the band, ~20x the seed spread). Do not drop it.

## How to judge these results

exp27's `lossfix` pin (`bc0b16b`) **does** include the mIoU re-basing `68b635f`, so unlike
exp30-vs-exp24, **both CE and mIoU are directly comparable here**:

| exp27 `lossfix`-s0 reference | value |
|---|---|
| best `eval/ce_mean` (mean t1-t4) | **0.68577** |
| final `eval/ce_mean` | 0.68676 |
| `eval/miou_final` | 0.44848 |
| `eval/miou_mean` | 0.43101 |

Published qband: **0.6853 +- 0.0007** mean t1-t4 val CE -> [0.6846, 0.6860]. `lossfix`-s0's
best sits inside it.

Expect the 10 seeds to cluster with best `ce_mean` ~0.685-0.686 and `miou_final` ~0.445-0.450.

**If a seed comes out materially BETTER than the band, distrust it.** That exact failure mode
has occurred here: `1a0b452` changed the resize default to `center_crop` and an arm landed
0.6693 — 0.016 "better" than the band, ~20x the seed spread — purely from the protocol change.
A result that beats the reference is evidence of a broken protocol until proven otherwise.
