#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

export KERASTUNER_TUNER_ID="chief"
export KERASTUNER_ORACLE_IP="127.0.0.1"
export KERASTUNER_ORACLE_PORT="8000"
python tune_keras.py &

for i in 0 1; do
    export KERASTUNER_TUNER_ID="tuner$i"
    export KERASTUNER_ORACLE_IP="127.0.0.1"
    export KERASTUNER_ORACLE_PORT="8000"
    python tune_keras.py &
done