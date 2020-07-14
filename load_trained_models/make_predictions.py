#!/usr/bin/env python
# coding: utf-8

# In[5]:


import tensorflow as tf 
import tensorflow_io as tfio
import tensorflow_probability as tfp

print('Tensorflow Version:')
print(tf.__version__)
print()
print('Tensorflow-Probability Version:')
print(tfp.__version__)
print()
print('Listing all GPU resources:')
print(tf.config.experimental.list_physical_devices('GPU'))
print()

import tensorflow.keras as keras
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import os
from tqdm import trange
import sys
import git
import importlib

mpl.rcParams['image.cmap'] = 'coolwarm'


# In[2]:


LAYER_NAME = 'encoder_5b'

FILTERS = 32
DATA_SIZE = 60000
PRIOR_MU = 0
PRIOR_SIGMA = 10

BATCH_SIZE = 128
EPOCHS = 200
VERBOSE = 2

N_PREDICTIONS = 100

ROOT_PATH = git.Repo("", search_parent_directories=True).git.rev_parse("--show-toplevel")
DATA_PATH = ROOT_PATH + "/data/"
SMALL_DATA_PATH = ROOT_PATH + "/load_trained_models" + "/data_small/"
LAYER_PATH = ROOT_PATH + "/layers/" + LAYER_NAME + "/"
SAVE_PATH = LAYER_PATH + LAYER_NAME + "_bayesian_model.h5"
PICKLE_PATH = LAYER_PATH + LAYER_NAME + '_hist.pkl'
MODEL_PATH = LAYER_PATH + LAYER_NAME + "_model"
IMAGE_PATH = ROOT_PATH + "/images/" + LAYER_NAME + "/"


# In[3]:


print("-" * 30)
print("Constructing model...")
print("-" * 30)
spec = importlib.util.spec_from_file_location(MODEL_PATH, MODEL_PATH + ".py")
ModelLoader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ModelLoader)
model = ModelLoader.make_model()
print(model.summary())


# In[4]:


model.load_weights(SAVE_PATH)
print("Model weights loaded successfully\n")


# In[5]:


n_test = 200
y_test = msks_test = np.load(SMALL_DATA_PATH + 'msks_test.npy')
imgs_test = np.load(SMALL_DATA_PATH + 'imgs_test.npy')

print("First " + str(n_test) + " test samples loaded\n")


# In[6]:


Xy_test = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(imgs_test),
                                tf.data.Dataset.from_tensor_slices(msks_test))
                             ).cache().batch(BATCH_SIZE).prefetch(8)


# In[7]:


prediction_size = list(msks_test.shape)
prediction_size.insert(0, N_PREDICTIONS)
prediction_test = np.zeros(prediction_size)


# In[8]:


print("Getting Monte Carlo samples of test predictions...")
for i in trange(N_PREDICTIONS):
    prediction_test[i] = model.predict(Xy_test)


# In[ ]:


for i in trange(0, 200, 20):
    plt.figure(dpi=100)
    plt.subplot(221)
    plt.title('Test Data T2-FLAIR')
    plt.imshow(imgs_test[i, :, :, 0], cmap='gray')

    plt.subplot(222)
    plt.title('Test Data T1')
    plt.imshow(imgs_test[i, :, :, 1], cmap='gray')

    plt.subplot(223)
    plt.title('Test Data T1-Contrast')
    plt.imshow(imgs_test[i, :, :, 2], cmap='gray')

    plt.subplot(224)
    plt.title('Test Data T2')
    plt.imshow(imgs_test[i, :, :, 3], cmap='gray')
    plt.tight_layout()
    plt.savefig(IMAGE_PATH +  'input_images_'+str(i).zfill(4)+'.png')

    plt.figure(dpi=200)

    plt.subplot(221)
    plt.title('\nPredicted Label Mean')
    plt.imshow(prediction_test.mean(0)[i, :, :, 0], interpolation='nearest')
    plt.colorbar()
    plt.clim(0, 1)

    plt.subplot(222)
    plt.title('\nPredicted Label Stddev')
    plt.imshow(prediction_test.std(0)[i, :, :, 0], interpolation='nearest')
    plt.clim(0, 0.2)
    plt.colorbar()

    plt.subplot(223)
    plt.title('True Label')
    plt.imshow(msks_test[i, :, :, 0], interpolation='nearest')
    plt.clim(0, 1)
    plt.colorbar()
    
    plt.subplot(224)
    plt.title('\nTruth-Prediction Discrepency')
    plt.imshow((msks_test[i, :, :, 0] - prediction_test.mean(0)[i, :, :, 0]), 
               interpolation='nearest')
    plt.clim(-1, 1)
    plt.colorbar()


    plt.tight_layout()

    plt.savefig(IMAGE_PATH + 'prediction_images_'+str(i).zfill(4)+'.png')


    plt.figure(dpi=100)

    plt.subplot(221)
    plt.title('\nPredicted Label 10th percentile')
    plt.imshow(np.percentile(prediction_test[:, i, :, :, 0], 10, axis=0), interpolation='nearest')
    plt.clim(0, 1)
    plt.colorbar()

    plt.subplot(222)
    plt.title('\nPredicted Label 50th percentile')
    plt.imshow(np.percentile(prediction_test[:, i, :, :, 0], 50, axis=0), interpolation='nearest')
    plt.clim(0, 1)
    plt.colorbar()

    plt.subplot(223)
    plt.title('Predicted Label 90th percentile')
    plt.imshow(np.percentile(prediction_test[:, i, :, :, 0], 90, axis=0), interpolation='nearest')
    plt.clim(0, 1)
    plt.colorbar()

    plt.subplot(224)
    plt.title('True Label')
    plt.imshow(msks_test[i, :, :, 0], interpolation='nearest')
    plt.clim(0, 1)
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(IMAGE_PATH + 'prediction_percentile_images_' + str(i).zfill(4)+'.png')


# In[10]:


history = pickle.load(open(PICKLE_PATH, 'rb'))


# In[11]:


plt.figure(dpi=100)
plt.semilogy(history['loss'], label='training loss (ELBO)')
plt.semilogy(history['val_loss'], label='testing loss')
plt.legend()
plt.savefig(IMAGE_PATH + 'training_history.png')


# In[12]:


for layer in model.layers:
    print(layer)
    weights = layer.get_weights()
    for w in weights:
        print(w.shape)


# In[ ]:




