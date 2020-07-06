#!/bin/bash

#SBATCH -A precisionhealth_project2
#SBATCH -p precisionhealth

#SBATCH --job-name=layer_2
#SBATCH --mail-user=hauthj@umich.edu
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem-per-cpu=7g
#SBATCH --time=24:00:00
#SBATCH --gpus=4



#SBATCH --output=/home/%u/tumor_segmentation_files/model_6_layer_2/%x-%j.log

# The application(s) to execute along with its input arguments and options:

python ./Layer_2_Bayesian.py
