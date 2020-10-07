#!/bin/bash

#SBATCH -A precisionhealth_owned1
#SBATCH -p precisionhealth

#SBATCH --job-name=all_layers_v3
#SBATCH --mail-user=hauthj@umich.edu
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem-per-cpu=7g
#SBATCH --time=12:00:00
#SBATCH --gpus=2
#SBATCH --output=%x-%j.log
# The application(s) to execute along with its input arguments and options:
export LAYER_NAME="deterministic"
echo "LAYER_NAME: "$LAYER_NAME
python ./deterministic_bayesian.py
