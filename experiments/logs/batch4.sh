#!/bin/zsh
S=$S
TR=/Users/vladosta/PycharmProjects/rwb-shipment-forecasting/data/train_solo_track.parquet
B='"feats":"v3","clip":true,"route_cat":false,"objective":"tweedie","rounds":[300,400,600]'
run() { $S/venv/bin/python -I $S/exp/exp2.py $TR "$1" 2>&1 | grep -E "RESULT|rows|Error|Traceback" ; }
run "{\"tag\":\"T_h1s\",$B,\"h1_struct\":true}"
run "{\"tag\":\"T_p11\",$B,\"tweedie_p\":1.1}"
run "{\"tag\":\"T_p15\",$B,\"tweedie_p\":1.5}"
run "{\"tag\":\"T_reg\",$B,\"leaves\":31,\"min_leaf\":1000,\"lr\":0.03,\"rounds\":[500,800,1100]}"
run "{\"tag\":\"T_perh\",$B,\"per_h\":true,\"rounds\":[150,250,400]}"
run "{\"tag\":\"T_seeds\",$B,\"seeds\":[1,2,3]}"
echo BATCH4 DONE
