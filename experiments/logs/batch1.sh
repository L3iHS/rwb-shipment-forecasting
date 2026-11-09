#!/bin/zsh
S=$S
TR=/Users/vladosta/PycharmProjects/rwb-shipment-forecasting/data/train_solo_track.parquet
run() { $S/venv/bin/python -I $S/exp/exp2.py $TR "$1" 2>&1 | grep -E "RESULT|rows|top imp|Error" ; }
run '{"tag":"A_v1"}'
run '{"tag":"B_v2","feats":"v2"}'
run '{"tag":"C_v2clip","feats":"v2","clip":true}'
run '{"tag":"D_v2clip_med","feats":"v2","clip":true,"level":"median"}'
run '{"tag":"E_v2clip_nocat","feats":"v2","clip":true,"route_cat":false}'
run '{"tag":"F_v2clip_allslots","feats":"v2","clip":true,"origin_slots":[0,47],"train_days":28}'
run '{"tag":"G_v2clip_70d","feats":"v2","clip":true,"train_days":70}'
echo BATCH1 DONE
