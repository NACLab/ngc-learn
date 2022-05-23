#!/bin/bash

################################################################################
# Run/simulate the GNCN-t1-FFM (or PCN)
# @author Alexander Ororbia
################################################################################

# Simulate PCN
echo "=> Running GNCN-t1-FFM model!"
python sim_train.py --config=gncn_t1_ffm/fit.cfg --gpu_id=0
python eval_model.py --config=gncn_t1_ffm/fit.cfg --gpu_id=0
