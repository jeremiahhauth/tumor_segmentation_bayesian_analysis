#!/bin/bash

#SBATCH -A precisionhealth_project2
#SBATCH -p precisionhealth

#SBATCH --job-name=make_predictions
#SBATCH --mail-user=hauthj@umich.edu
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=7g
#SBATCH --time=1:00:00
#SBATCH --gpus=1



#SBATCH --output=%x-%j.log

# The application(s) to execute along with its input arguments and options:
export LAYER_NAME="make_predictions"
echo "LAYER_NAME: "$LAYER_NAME
python ./make_predictions.py
