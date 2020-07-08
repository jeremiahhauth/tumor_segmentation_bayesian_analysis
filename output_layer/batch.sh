#!/bin/bash
export LAYER_NAME="output_layer"
echo "LAYER_NAME: "$LAYER_NAME

#SBATCH -A precisionhealth_project2
#SBATCH -p precisionhealth

#SBATCH --job-name=$LAYER_NAME
#SBATCH --mail-user=hauthj@umich.edu
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem-per-cpu=7g
#SBATCH --time=24:00:00
#SBATCH --gpus=3

#SBATCH --output=/home/%u/tumor_segmentation_files/$LAYER_NAME/%x-%j.log

# The application(s) to execute along with its input arguments and options:

python ./${LAYER_NAME}"_bayesian.py"
