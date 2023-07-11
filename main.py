# %%
import cv2
import os
import glob
import warnings
import scipy.misc
import numpy as np
# # import nibabel as nib
import SimpleITK as sitk
from scipy import ndimage
import matplotlib.pyplot as plt
# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
from matplotlib.widgets import Slider

# import keras.api._v2.keras as keras
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Activation, Conv2DTranspose, concatenate, Dropout
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from scipy.spatial.distance import directed_hausdorff

# %%
from preprocessing import preprocessing

print('-'*30)
print('Loading and preprocessing training data...')
print('-'*30)

# \\files.auckland.ac.nz\research\resmed202100086-tws ----> Address for raw data
scans_path = 'D:/MRI - Tairawhiti'
# filename_labels = ['R_tibia_15A', 'R_tibia_16A', 'R_tibia_4A']
filename_labels = ['R_tibia_15A']
imgs_train, imgs_mask_train, raw_data  = preprocessing(scans_path, filename_labels)

#Sample size (temp)
# imgs_train = imgs_train[:10, :, :, :]
# imgs_mask_train = imgs_mask_train[:10, :, :, :]

# Normalization
imgs_mask_train = imgs_mask_train.astype('float32')
imgs_train = imgs_train.astype('float32')
# imgs_mask_train /= 255.  # scale masks to [0, 1]
# imgs_train /= 255.  # scale masks to [0, 1]
print("\n")
print('Final Training Image Input Shape: ', imgs_train.shape)
print('Final Training Mask Input Shape: ', imgs_mask_train.shape)

print('-'*30)
print('Completed Preprocessing Stage!')
print('-'*30)