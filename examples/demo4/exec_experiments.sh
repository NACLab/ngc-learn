#!/bin/bash

################################################################################
# Run a set of experiments and analyze each sparse coding model
# @author Alexander Ororbia
################################################################################

echo "=> Running GNCN-t1-SC model w/ Cauchy prior!"
python sim_train.py --config=sc_cauchy/fit.cfg --gpu_id=0
python viz_filters.py --model_fname=sc_cauchy/model0.ngc --output_dir=sc_cauchy/

echo "=> Running GNCN-t1-SC model learned through ISTA!"
python sim_train.py --config=sc_ista/fit.cfg --gpu_id=0
python viz_filters.py --model_fname=sc_ista/model0.ngc --output_dir=sc_ista/
