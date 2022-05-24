#!/bin/bash

################################################################################
# Run a set of experiments and analyze each generative model
# @author Alexander Ororbia
################################################################################

# Simulate GNCN-t1/Rao
echo "=> Running GNCN-t1/Rao experiment!"
python sim_train.py --config=gncn_t1/fit.cfg --gpu_id=0
python extract_latents.py --config=gncn_t1/analyze.cfg --gpu_id=0
python fit_gmm.py --config=gncn_t1/analyze.cfg --gpu_id=0
python eval_logpx.py --config=gncn_t1/analyze.cfg --gpu_id=0

# Simulate GNCN-t1-Sigma/Friston
echo "=> Running GNCN-t1-Sigma/Friston experiment!"
python sim_train.py --config=gncn_t1_sigma/fit.cfg --gpu_id=0
python extract_latents.py --config=gncn_t1_sigma/analyze.cfg --gpu_id=0
python fit_gmm.py --config=gncn_t1_sigma/analyze.cfg --gpu_id=0
python eval_logpx.py --config=gncn_t1_sigma/analyze.cfg --gpu_id=0

# Simulate GNCN-PDH
echo "=> Running GNCN-PDH experiment!"
python sim_train.py --config=gncn_pdh/fit.cfg --gpu_id=0
python extract_latents.py --config=gncn_pdh/analyze.cfg --gpu_id=0
python fit_gmm.py --config=gncn_pdh/analyze.cfg --gpu_id=0
python eval_logpx.py --config=gncn_pdh/analyze.cfg --gpu_id=0

