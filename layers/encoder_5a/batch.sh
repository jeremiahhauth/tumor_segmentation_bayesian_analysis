#!/bin/bash

#SBATCH -A precisionhealth_project2
#SBATCH -p precisionhealth

#SBATCH --job-name=encoder_5a
#SBATCH --mail-user=hauthj@umich.edu
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem-per-cpu=7g
#SBATCH --time=24:00:00
#SBATCH --gpus=3



#SBATCH --output=$SLURM_SUBMIT_DIR/%x-%j.log

# The application(s) to execute along with its input arguments and options:
export LAYER_NAME="encoder_5a"
echo "LAYER_NAME: "$LAYER_NAME
python ./encoder_5a_bayesian.py
