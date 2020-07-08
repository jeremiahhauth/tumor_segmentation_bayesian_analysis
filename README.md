# tumor_segmentation_bayesian_analysis
Conducting Bayesian inference on various subsets of parameters in the U-NET architecture for the application of tumor segmentation

## Layers
Models are trained by doing Flipout MFVI on each trainable layer. Trainable layers follow the following naming scheme (assigned to environment variable `LAYER_NAME`):

- `encoder_1a`
- `encoder_1b`
- `encoder_2a`
- `encoder_2b`
- `encoder_3a`
- `encoder_3b`
- `encoder_4a`
- `encoder_4b`
- `encoder_5a`
- `encoder_5b`
- `decoder_4a`
- `decoder_4b`
- `decoder_3a`
- `decoder_3b`
- `decoder_2a`
- `decoder_2b`
- `decoder_1a`
- `decoder_1b`
- `output_layer`

Each of these layers as well as the fully bayesian model has a directory with this name. Additionally, a model is trained in which Flipout MFVI is conducted on all layer simuleniously under the directory:

- `all_layers`

Within a layer directory, there are several files:
- `batch.sh`  slurm script to run model training
- `LAYER_NAME_bayesian.py`  main file with training parameters
- `LAYER_NAME_model.py` a function called by `LAYER_NAME_bayesian.py` and returns a compiled, untrained model.
- `LAYER_NAME-######.log` command line output from running `LAYER_NAME_bayesian.py` via slurm
- `LAYER_NAME_bayesian_model.h5` saved weights from a fully trained model
- `LAYER_NAME_hist.pkl` python dict that stores metrics from training history



## Load_Trained_Models
Included here is a script [make_predictions.ipynb] that will load and plot trained models

## Images
Plots of predictions with mean, standard deviation, and percentiles are stores are stored under images.

Below is a diagram of the U-NET architecture with all deterministic layers.

<img src="https://github.com/jeremiahhauth/tumor_segmentation_bayesian_analysis/blob/master/images/deterministic_model.png" width="400" class="center">
