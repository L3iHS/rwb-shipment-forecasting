#!/bin/zsh
S=$S
TR=/Users/vladosta/PycharmProjects/rwb-shipment-forecasting/data/train_solo_track.parquet
B='"feats":"v2","clip":true,"route_cat":false'
run() { $S/venv/bin/python -I $S/exp/exp2.py $TR "$1" 2>&1 | grep -E "RESULT|rows|Error|Traceback" ; }
run "{\"tag\":\"E2_huber\",$B,\"objective\":\"huber\"}"
run "{\"tag\":\"E2_tweedie\",$B,\"objective\":\"tweedie\"}"
run "{\"tag\":\"E2_reg\",$B,\"leaves\":31,\"min_leaf\":1000,\"lr\":0.03,\"rounds\":[400,700,1000]}"
run "{\"tag\":\"E2_perh\",$B,\"per_h\":true,\"rounds\":[150,250,400]}"
run "{\"tag\":\"E2_wide\",$B,\"origin_slots\":[10,32]}"
run "{\"tag\":\"X_xgb\",$B,\"algo\":\"xgb\",\"rounds\":[200,400,600]}"
run "{\"tag\":\"C_cat\",$B,\"algo\":\"cat\",\"rounds\":[400,800]}"
echo BATCH2 DONE
