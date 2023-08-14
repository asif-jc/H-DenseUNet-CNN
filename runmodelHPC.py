#Libraries
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
from keras.models import Model
from scipy.spatial.distance import directed_hausdorff

def ReadInDatasets(folder_path):
    files = os.listdir(folder_path)
    num_files = len(files)
    result_array = np.empty((num_files, 512, 512))

    for i, file in enumerate(files):
        file_path = os.path.join(folder_path, file)
        with open(file_path, 'r') as f:
            # Read the contents of the text file
            content = f.read()

            # Convert the content into a 2D NumPy array of floats
            array_2d = np.array([list(map(float, line.split(','))) for line in content.splitlines()])

            # Store the 2D array in the result array
            result_array[i] = array_2d

    return result_array

def get_unet(scale = 0.5, dropout_rate = 0.4):
    inputs = keras.Input((512,512,1))

    # Encoding Path of the UNet (32-64-128-256-512)
    conv1   = Conv2D(32*scale, (3, 3), padding="same", activation='relu')(inputs)
    # conv1 = Conv2D(32*scale, 3, activation='relu', padding='same')(inputs)
    drop1   = Dropout(rate=dropout_rate)(conv1, training=True)
    max1    = MaxPooling2D((2, 2))(drop1)

    conv2   = Conv2D(64*scale, (3, 3), padding="same", activation='relu')(max1)
    drop2   = Dropout(rate=dropout_rate)(conv2, training=True)
    max2    = MaxPooling2D((2, 2))(drop2)

    conv3   = Conv2D(128*scale, (3, 3), padding="same", activation='relu')(max2)
    drop3   = Dropout(rate=dropout_rate)(conv3, training=True)
    max3    = MaxPooling2D((2, 2))(drop3)

    conv4   = Conv2D(256*scale, (3, 3), padding="same", activation='relu')(max3)
    drop4   = Dropout(rate=dropout_rate)(conv4, training=True)
    max4    = MaxPooling2D((2, 2))(drop4)

    lat     = Conv2D(512*scale, (3, 3), padding="same", activation='relu')(max4)
    drop5   = Dropout(rate=dropout_rate)(lat, training=True)

    # Decoding Path of the UNet
    up1     = UpSampling2D((2, 2))(drop5)
    concat1 = concatenate([conv4, up1], axis=-1)
    conv5   = Conv2D(256*scale, (3, 3), padding="same", activation='relu')(concat1)
    drop6   = Dropout(rate=dropout_rate)(conv5, training=True)
    
    up2     = UpSampling2D((2, 2))(drop6)
    concat2 = concatenate([conv3, up2], axis=-1)
    conv6   = Conv2D(128*scale, (3, 3), padding="same", activation='relu')(concat2)
    drop7   = Dropout(rate=dropout_rate)(conv6, training=True)
    
    up3     = UpSampling2D((2, 2))(drop7)
    concat3 = concatenate([conv2, up3], axis=-1)
    conv7   = Conv2D(64*scale, (3, 3), padding="same", activation='relu')(concat3)
    drop8   = Dropout(rate=dropout_rate)(conv7, training=True)

    up4     = UpSampling2D((2, 2))(drop8)
    concat4 = concatenate([conv1, up4], axis=-1)
    conv8   = Conv2D(32*scale, (3, 3), padding="same", activation='relu')(concat4)
    drop9   = Dropout(rate=dropout_rate)(conv8, training=True)
    
    outputs = Conv2D(1, (1, 1), activation="softmax")(drop9)

    model   = Model(inputs, outputs)

    return model

imgs_train = ReadInDatasets('Data_Tibia')
imgs_mask_train = ReadInDatasets('Masks_Tibia')
imgs_mask_train = np.expand_dims(imgs_mask_train, axis=-1)
imgs_train = np.expand_dims(imgs_train, axis=-1)

# Image Parameters
IMAGE_WDITH = 512
IMAGE_HEIGHT = 512
IMAGE_CHANNELS = 1

# Training, Testing and Validation Parameters
# TRAINING_VOLUMES = [0, 1, 2, 3, 4, 5, 6, 7, 8]
# VALIDATION_VOLUMES = [9]

# Hyperparameters
N_CLASSES = 2
N_INPUT_CHANNELS = 1
# PATCH_SIZE = (32, 32)
# PATCH_STRIDE = (32, 32)

# # Data Preparation Parameters
# CONTENT_THRESHOLD = 0.3 # To Get Rid of Useless Information in the Image

# Training Parameters
N_EPOCHS = 200
BATCH_SIZE = 32
PATIENCE = 3
MODEL_FNAME_PATTERN = 'model.h5'
OPTIMISER = 'Adam'
# LOSS = 'categorical_crossentropy'
LOSS = 'binary_crossentropy'
dropout_rate = 0.40


print('-'*30)
print('Loading and preprocessing train data...')
print('-'*30)

images_train, images_val, labels_train, labels_val = train_test_split(imgs_train, imgs_mask_train, test_size=0.2, random_state=0)
print('Training Image Input Shape: ', images_train.shape)
print('Training Mask Input Shape: ', labels_train.shape)
print('Validation Image Input Shape: ', images_val.shape)
print('Validation Mask Input Shape: ', labels_val.shape)


print('-'*30)
print('Creating and compiling model...')
print('-'*30)

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=PATIENCE), # early stopping
    tf.keras.callbacks.ModelCheckpoint(filepath=MODEL_FNAME_PATTERN, save_best_only=True) # save the best based on validation
]

unet = get_unet()
unet.compile(optimizer=OPTIMISER, loss=LOSS)
unet.fit(
    x=images_train, 
    y=labels_train,
    validation_data=(images_val, labels_val),
    batch_size=BATCH_SIZE,
    epochs=N_EPOCHS,
    callbacks=my_callbacks,
    verbose=1)

unet.summary()

unet = get_unet()
unet.compile(optimizer=OPTIMISER, loss=LOSS)
unet.load_weights('model.h5')

testing_scans_processed = images_train[100]
testing_masks_processed = labels_train[100]
testing_scans_processed = np.reshape(testing_scans_processed, (1, 512, 512, 1))
testing_masks_processed = np.reshape(testing_masks_processed, (1, 512, 512, 1))

# testing_labels_processed = tf.keras.utils.to_categorical(testing_masks_processed, num_classes=2, dtype='float32')

print('Testing Image Input Shape: ',testing_scans_processed.shape)
print('Testing Mask Input Shape:',testing_masks_processed.shape)

prediction = unet.predict(x=testing_scans_processed)
# prediction = np.argmax(prediction, axis=3)
# prediction = np.reshape(prediction[0], (512, 512))

# plt.imshow(prediction, cmap='gray')
# plt.show()
np.savetxt('output.txt', prediction.reshape(prediction.shape[0], -1))
