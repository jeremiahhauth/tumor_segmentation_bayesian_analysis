#!/bin/bash

#SBATCH -A precisionhealth_project2
#SBATCH -p precisionhealth

#SBATCH --job-name=layer_3
#SBATCH --mail-user=hauthj@umich.edu
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem-per-cpu=7g
#SBATCH --time=24:00:00
#SBATCH --gpus=3



#SBATCH --output=/home/%u/tumor_segmentation_bayesian_analysis/layer_3/%x-%j.log

# The application(s) to execute along with its input arguments and options:

python ./Layer_3_Bayesian.py
