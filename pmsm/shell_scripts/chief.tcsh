#!/bin/tcsh

setenv KERASTUNER_TUNER_ID "chief"
setenv KERASTUNER_ORACLE_IP "127.0.0.1"
setenv KERASTUNER_ORACLE_PORT "8000"
python tune_keras.py &