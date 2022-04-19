#!/bin/bash

# run set of experiments and analyze each generative model

# Simulate GNCN-PDH
# python sim_train.py --config=fit_gncn_pdf.cfg --gpu_id=0
# python extract_latents.py --config=analyze_gncn_pdh.cfg --gpu_id=0
# python fit_gmm.py --config=analyze_gncn_pdh.cfg --gpu_id=0
# python eval_logpx.py --config=analyze_gncn_pdh.cfg --gpu_id=0
#
# # Simulate GNCN-t1-Sigma/Friston
# python sim_train.py --config=fit_gncnt1_sigma.cfg --gpu_id=0
# python extract_latents.py --config=analyze_gncnt1_sigma.cfg --gpu_id=0
# python fit_gmm.py --config=analyze_gncnt1_sigma.cfg --gpu_id=0
# python eval_logpx.py --config=analyze_gncnt1_sigma.cfg --gpu_id=0
#
# # Simulate GNCN-t1/Rao
# python sim_train.py --config=fit_gncnt1.cfg --gpu_id=0
python extract_latents.py --config=analyze_gncnt1.cfg --gpu_id=0
python fit_gmm.py --config=analyze_gncnt1.cfg --gpu_id=0
python eval_logpx.py --config=analyze_gncnt1.cfg --gpu_id=0
