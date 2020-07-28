#!/bin/bash

#SBATCH -A precisionhealth_owned1
#SBATCH -p precisionhealth

#SBATCH --job-name=encoder_1a_v2
#SBATCH --mail-user=hauthj@umich.edu
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=7g
#SBATCH --time=36:00:00
#SBATCH --gpus=8



#SBATCH --output=%x-%j.log

# The application(s) to execute along with its input arguments and options:
export LAYER_NAME="encoder_1a_v2"
echo "LAYER_NAME: "$LAYER_NAME
python ./encoder_1a_v2_bayesian.py
