# Throughput benchmarks

Regenerate anytime: `uv run python -m canvit_pytorch_rl.tools.bench` → `runs/bench.json`
(records host, git hash, torch version). Summary below from crockett
(RTX 4090 24G, torch 2.11.0+cu128, bf16 autocast, 128² glimpses, 512² scenes,
frozen weights), git `0408107`, 2026-06-11.

One glimpse = one CanViT forward (the budget unit). "backward" = gradients
flowing through grid_sample + the frozen model to the viewpoint (pathwise
policy-gradient setting), backward included.

| canvas | mode | peak glimpses/s | saturation | peak VRAM at B=16 / B=64 |
|---|---|---:|---|---|
| 64² | infer | ~430 | B≥16 | 2.6G / 9.4G (B=128: 18.4G) |
| 64² | backward | ~150 | B≥8 | 4.7G / 17.1G (B=128: OOM) |
| 32² | infer | ~1,350 | B≥16 | 1.0G / 2.8G |
| 32² | backward | ~470 | B≥8 | 2.0G / 6.4G |

What this means for experiment costing (c64 unless said otherwise):

- Full-val T=2 eval (4k forwards + scoring): ~20 s. T=21 (42k forwards): ~2.5 min.
- Best-of-17 t1 oracle on val (36k forwards + 17× scoring): ~2.5 min.
- One epoch of offline critic training over the saved 20210-image train
  candidate dataset = 20210 t0 forwards (no model backward): ~47 s + head cost.
- The Codex-era 16.4M-forward critic budget ≈ 10.6 h; the "strict 1M" budget
  ≈ 39 min — both at inference rate (their candidate forwards carried no
  model backward either).
- c32 is ~3.2× cheaper than c64 across the board; paper Table 5 shows policy
  ORDERING is preserved at c32 → fast-iteration rung.
