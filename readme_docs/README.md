# readme_docs

Extended documentation linked from the top-level [README](../README.md). The
README describes the repository; these documents describe *procedures* — specific
training campaigns, how to launch them, and how to judge their results.

| document | contents |
|---|---|
| [`policy_on_own_model.md`](policy_on_own_model.md) | Training the ADE20K viewpoint policy on a backbone and probe you trained yourself: the two halves a policy run needs, the three settings that fail silently if mismatched, and the ready-made launcher. |
| [`verification_runs.md`](verification_runs.md) | The exp32–exp35 campaign: four groups of runs that verify pretraining, ImageNet-1k finetuning, ADE20K probing and viewpoint-policy training at full scale against earlier reference results. |

## Assets

| file | produced by |
|---|---|
| `ComparisonPoliciesADE20K.png` | `scripts/plot_policy_comparison.py` — the learned Viewpoint-Q policy against the open-loop baselines over t = 0..4, with the paper's Table 4 rows as dashed references |
| `_policy_comparison_data.json` | the measure stage of that same script (cached evaluations, so re-styling the figure costs no GPU) |
