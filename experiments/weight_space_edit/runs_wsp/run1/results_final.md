## Results (test unsealed once)

| arm | sel TPR@5% | sel AUROC | test TPR@5% | test FPR@5% | test TPR@1% | test FPR@1% | test macro-AUROC | edit size |
|---|---|---|---|---|---|---|---|---|
| original (θ0) | 0.317 | 0.789 | 0.503 | 0.083 | 0.406 | 0.019 | 0.791 | 0.000 |
| **wsp_claude** (c15) | 0.378 | 0.779 | 0.500 | 0.075 | 0.400 | 0.017 | 0.790 | 1.069 |
| gd_finetune | 0.394 | 0.802 | 0.519 | 0.097 | 0.397 | 0.011 | 0.809 | 1.774 |
| refit_all_dev | 0.406 | 0.794 | **0.533** | 0.083 | 0.406 | 0.011 | 0.806 | 0.622 |
| random_edit_control | 0.367 | 0.784 | 0.486 | 0.072 | 0.369 | 0.019 | 0.787 | 1.069 |

Paired differences, bootstrap over the 6 test families (mean [95% interval]), threshold from calibration at 5% FPR:

| comparison | Δ test TPR@5% | Δ test FPR@5% | Δ test macro-AUROC |
|---|---|---|---|
| wsp_claude − original | −0.003 [−0.039, +0.025] | −0.008 [−0.019, +0.006] | −0.002 [−0.018, +0.018] |
| wsp_claude − gd_finetune | −0.020 [−0.081, +0.044] | −0.023 [−0.061, +0.014] | −0.019 [−0.050, +0.018] |
| wsp_claude − refit_all_dev | −0.034 [−0.069, −0.003] | −0.009 [−0.050, +0.025] | −0.016 [−0.046, +0.024] |
| wsp_claude − random_edit_control | +0.013 [−0.044, +0.075] | +0.003 [−0.011, +0.017] | +0.002 [−0.011, +0.018] |
| gd_finetune − original | +0.016 [−0.039, +0.067] | +0.015 [−0.017, +0.053] | +0.017 [−0.001, +0.034] |

Per-family test AUROC:

| family | original | wsp_claude | gd_finetune | refit_all_dev | random_edit |
|---|---|---|---|---|---|
| among | 0.635 | **0.677** | 0.614 | 0.603 | 0.661 |
| harm | 0.935 | 0.921 | 0.952 | 0.946 | 0.934 |
| medqa | 0.743 | 0.713 | 0.792 | 0.780 | 0.731 |
| model | 0.931 | 0.921 | 0.939 | 0.944 | 0.923 |
| sandbagging | 0.857 | 0.846 | 0.873 | 0.876 | 0.852 |
| werewolf | 0.648 | 0.661 | 0.682 | 0.687 | 0.625 |

