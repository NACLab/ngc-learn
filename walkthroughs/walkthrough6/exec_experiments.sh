#!/bin/bash

################################################################################
# Run a set of experiments and analyze the deep hierarchical ISTA model
# @author Alexander Ororbia
################################################################################

# Simulate deep ISTA
echo "=> Running Harmonium model!"
python sim_train.py --config=rbm/fit.cfg --gpu_id=0
python sample_rbm.py --model_fname=rbm/model0.ngc --output_dir=rbm/
python viz_filters.py --model_fname=rbm/model0.ngc --output_dir=rbm/
