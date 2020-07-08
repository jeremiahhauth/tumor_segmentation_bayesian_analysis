#!/bin/bash

#SBATCH -A precisionhealth_project2
#SBATCH -p precisionhealth

#SBATCH --job-name=output_layer
#SBATCH --mail-user=hauthj@umich.edu
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem-per-cpu=7g
#SBATCH --time=24:00:00
#SBATCH --gpus=3



#SBATCH --output=/home/%u/tumor_segmentation_bayesian_analysis/output_layer/%x-%j.log

# The application(s) to execute along with its input arguments and options:
export LAYER_NAME="output_layer"
echo "LAYER_NAME: "$LAYER_NAME
python ./output_layer_bayesian.py
