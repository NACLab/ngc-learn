#!/bin/bash

################################################################################
# Run a set of experiments and analyze the deep hierarchical ISTA model
# @author Alexander Ororbia
################################################################################

# Simulate deep ISTA
echo "=> Running GNCN-t1-ISTA model!"
python sim_train.py --config=sc_face/fit.cfg --gpu_id=0
python viz_filters.py --model_fname=sc_face/model0.ngc --output_dir=sc_face/ --viz_encoder=True
