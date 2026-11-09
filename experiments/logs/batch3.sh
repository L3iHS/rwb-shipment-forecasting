#!/bin/zsh
S=$S
TR=/Users/vladosta/PycharmProjects/rwb-shipment-forecasting/data/train_solo_track.parquet
B='"feats":"v3","clip":true,"route_cat":false'
run() { $S/venv/bin/python -I $S/exp/exp2.py $TR "$1" 2>&1 | grep -E "RESULT|rows|top imp|Error|Traceback" ; }
run "{\"tag\":\"V3_l2\",$B}"
run "{\"tag\":\"V3_tw\",$B,\"objective\":\"tweedie\"}"
run "{\"tag\":\"V3_xgb\",$B,\"algo\":\"xgb\",\"rounds\":[200,400,600]}"
run "{\"tag\":\"V3_cat\",$B,\"algo\":\"cat\",\"rounds\":[400,800]}"
echo BATCH3 DONE
