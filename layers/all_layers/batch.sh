#!/bin/bash

#SBATCH -A precisionhealth_owned1
#SBATCH -p precisionhealth

#SBATCH --job-name=all_layers
#SBATCH --mail-user=hauthj@umich.edu
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem-per-cpu=7g
#SBATCH --time=72:00:00
#SBATCH --gpus=8

#SBATCH --output=%x-%j.log

# The application(s) to execute along with its input arguments and options:
export LAYER_NAME="all_layers"
echo "LAYER_NAME: "$LAYER_NAME
python ./all_layers_bayesian.py
