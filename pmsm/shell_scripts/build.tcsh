#!/bin/tcsh

docker build -t dergbsl1078:5043/ngc_tf:hyper_test .
docker push dergbsl1078:5043/ngc_tf:hyper_test
singularity pull ngc_tf_hyper_test.sif docker://dergbsl1078:5043/ngc_tf:hyper_test

# salloc -p a100 --gres=gpu:4
# srun --pty /bin/bash
# singularity run --nv ngc_tf_hyper_test.sif
