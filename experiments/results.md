# Результаты на DEV (14 срезов в 10:30, 11.10–24.10)

Метрика платформы WAPE + |Relative Bias|, среднее по срезам; меньше — лучше. Для бустингов показано лучшее число деревьев.
Собрано скриптом `collect_results.py` из `logs/`.

| Тег | DEV | std | WAPE | bias | субботы | деревья | конфиг |
|---|---:|---:|---:|---:|---:|---|---|
| T_h1s | 0.3283 | 0.0113 | 0.3165 | -0.0003 | 0.3300 | 600 деревьев | `{"tag":"T_h1s","feats":"v3","clip":true,"route_cat":false,"objective":"tweedie","rounds":[300,400,600],"h1_struct":true}` |
| V3_tw | 0.3296 | 0.0130 | 0.3172 | -0.0005 | 0.3325 | 300 деревьев | `{"tag":"V3_tw","feats":"v3","clip":true,"route_cat":false,"objective":"tweedie"}` |
| V3_xgb | 0.3321 | 0.0150 | 0.3179 | +0.0037 | 0.3403 | 200 деревьев | `{"tag":"V3_xgb","feats":"v3","clip":true,"route_cat":false,"algo":"xgb","rounds":[200,400,600]}` |
| V3_l2 | 0.3326 | 0.0140 | 0.3177 | +0.0040 | 0.3352 | 300 деревьев | `{"tag":"V3_l2","feats":"v3","clip":true,"route_cat":false}` |
| V3_cat | 0.3327 | 0.0132 | 0.3177 | +0.0006 | 0.3361 | 400 деревьев | `{"tag":"V3_cat","feats":"v3","clip":true,"route_cat":false,"algo":"cat","rounds":[400,800]}` |
| E2_tweedie | 0.3336 | 0.0125 | 0.3194 | -0.0058 | 0.3366 | 600 деревьев | `{"tag":"E2_tweedie","feats":"v2","clip":true,"route_cat":false,"objective":"tweedie"}` |
| E_v2clip_nocat | 0.3339 | 0.0132 | 0.3200 | -0.0046 | 0.3404 | 400 деревьев | `{"tag":"E_v2clip_nocat","feats":"v2","clip":true,"route_cat":false}` |
| E2_reg | 0.3344 | 0.0126 | 0.3197 | -0.0045 | 0.3398 | 1000 деревьев | `{"tag":"E2_reg","feats":"v2","clip":true,"route_cat":false,"leaves":31,"min_leaf":1000,"lr":0.03,"rounds":[400,700,1000]}` |
| F_v2clip_allslots | 0.3353 | 0.0148 | 0.3206 | +0.0002 | 0.3430 | 300 деревьев | `{"tag":"F_v2clip_allslots","feats":"v2","clip":true,"origin_slots":[0,47],"train_days":28}` |
| E2_huber | 0.3358 | 0.0115 | 0.3186 | -0.0131 | 0.3385 | 600 деревьев | `{"tag":"E2_huber","feats":"v2","clip":true,"route_cat":false,"objective":"huber"}` |
| D_v2clip_med | 0.3364 | 0.0146 | 0.3214 | -0.0029 | 0.3441 | 300 деревьев | `{"tag":"D_v2clip_med","feats":"v2","clip":true,"level":"median"}` |
| B_v2 | 0.3371 | 0.0142 | 0.3224 | -0.0017 | 0.3445 | 200 деревьев | `{"tag":"B_v2","feats":"v2"}` |
| C_v2clip | 0.3371 | 0.0145 | 0.3224 | -0.0019 | 0.3456 | 200 деревьев | `{"tag":"C_v2clip","feats":"v2","clip":true}` |
| G_v2clip_70d | 0.3377 | 0.0144 | 0.3219 | -0.0034 | 0.3467 | 200 деревьев | `{"tag":"G_v2clip_70d","feats":"v2","clip":true,"train_days":70}` |
| N2_huber | 0.3399 |  | 0.3214 | -0.0081 |  |  | `N-HiTS, см. logs/nhits1.log` |
| A_v1 | 0.3402 | 0.0155 | 0.3232 | -0.0055 | 0.3521 | 200 деревьев | `{"tag":"A_v1"}` |
| N1_mae | 0.3437 |  | 0.3230 | -0.0066 |  |  | `N-HiTS, см. logs/nhits1.log` |
| N6_huber_cur | 0.3442 |  | 0.3199 | -0.0038 |  |  | `N-HiTS, см. logs/nhits2.log` |
| N4_huber_s2 | 0.3469 |  | 0.3220 | -0.0123 |  |  | `N-HiTS, см. logs/nhits2.log` |
| N7_huber_s3 | 0.3635 |  | 0.3232 | -0.0403 |  |  | `N-HiTS, см. logs/nhits2.log` |
| N5_mse | 0.3639 |  | 0.3286 | +0.0354 |  |  | `N-HiTS, см. logs/nhits2.log` |
| N3_mae_in672 | 0.3950 |  | 0.3279 | -0.0671 |  |  | `N-HiTS, см. logs/nhits1.log` |
