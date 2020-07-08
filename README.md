# tumor_segmentation_bayesian_analysis
Conducting Bayesian inference on various subsets of parameters in the U-NET architecture for the application of tumor segmentation

## Layers
Trainable layers follow the following naming scheme (refer to the network diagram for full layer interactions):

- encoder_1a
- encoder_1b
- encoder_2a
- encoder_2b
- encoder_3a
- encoder_3b
- encoder_4a
- encoder_4b
- encoder_5a
- encoder_5b
- decoder_4a
- decoder_4b
- decoder_3a
- decoder_3b
- decoder_2a
- decoder_2b
- decoder_1a
- decoder_1b
- output_layer

## Images
Plots of predictions with mean, standard deviation, and percentiles are stores are stored under images.

Below is a diagram of the U-NET architecture with all deterministic layers.
<img src="https://github.com/jeremiahhauth/tumor_segmentation_bayesian_analysis/blob/master/images/deterministic_model.png" width="400" class="center">
