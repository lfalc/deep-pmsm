#!/bin/bash

for i in 0 1
do
    export KERASTUNER_TUNER_ID="tuner$i"
    export KERASTUNER_ORACLE_IP="127.0.0.1"
    export KERASTUNER_ORACLE_PORT="8000"
    python tune_keras.py &
done