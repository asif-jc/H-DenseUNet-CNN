# Libraries

# Base Functionalities
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import numpy as np
import cv2
import time
from random import randint
import SimpleITK as sitk
import pandas as pd
import sys

# CNN Libraries
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Activation, Conv2DTranspose, concatenate
from keras.layers.core import Dropout
import tensorflow as tf
# from tensorflow.keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Visualisation Libraries
import plotly.graph_objects as go
import pydicom
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Mask Creation Libraries
import trimesh
# import pyvista as pv
from PIL import Image
# from stl import mesh
from skimage import measure
from skimage.transform import resize
from plyfile import PlyData
from pyntcloud import PyntCloud

# Viewer
from scipy import ndimage as ndi
from skimage import (exposure, feature, filters, io, measure,
                      morphology, restoration, segmentation, transform,
                      util)
# import napari 
from matplotlib.widgets import Slider



# Helper Function
def flatten_3d_to_2d(array_3d):
    # Get the dimensions of the 3D array
    depth, height, width = array_3d.shape
    
    # Reshape the 3D array to a 2D array
    array_2d = np.reshape(array_3d, (depth, height * width))
    
    return array_2d
# Helper Function
def flatten_2d_array(arr):
    flattened = []
    for row in arr:
        flattened.extend(row)
    return flattened
# Helper Function
# Read in entire scan of single patient
# folders = [f for f in os.listdir('MRI Scans - Tairawhiti') if os.path.isdir(os.path.join('MRI Scans - Tairawhiti', f))]
def ListFolders(directory):
    folder_names = []
    for root, dirs, files in os.walk(directory):
        for folder in dirs:
            folder_names.append(folder)
    return folder_names
# Helper Function
def read_dicom_files(directory):
    dicom_files = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filename.endswith('.dcm'):
            try:
                dicom_file = pydicom.dcmread(filepath)
                dicom_files.append(dicom_file)
            except pydicom.errors.InvalidDicomError:
                print(f"Skipping file: {filename}. It is not a valid DICOM file.")
    return dicom_files
# Helper Function
def get_ram_usage(variable, variable_name):
    size_in_bytes = sys.getsizeof(variable)
    size_in_kb = size_in_bytes / 1024
    size_in_mb = size_in_kb / 1024
    size_in_gb = size_in_mb / 1024
    message = "Memory usage of %s: %d %s." % (variable_name, size_in_mb, 'MB')
    print(message)



def TrainingMRIScans(scans_path):

    folders = ListFolders(scans_path)
    scan_pixel_data = []
    scan_coordinate_data = []

    # Pixel Data
    for paitent in folders:
        single_scan_pixel_data = []
        # single_scan_coord_data = []
        single_paitent_scans_path =  scans_path + '/{}'.format(paitent)
        dicom_files = read_dicom_files(single_paitent_scans_path)
        # D:\MRI - Tairawhiti\AutoBind_WaterWATER_450
        for i in range (len(dicom_files)):
            single_scan_pixel_data.append(dicom_files[i].pixel_array)
            # single_scan_coord_data.append(dicom_files[i].ImagePositionPatient)
        scan_pixel_data.append(single_scan_pixel_data)
        # scan_coordinate_data.append(single_scan_pixel_data)

    training_scans = flatten_2d_array(scan_pixel_data)
    training_scans = np.array(training_scans)
    #TEMP
    training_scans = training_scans[0:1015]

    # Coordinate Data
    for paitent in folders:
        single_paitent_scans_path =  scans_path + '/{}'.format(paitent)
        for i in range (len(dicom_files)):
            scan_coordinate_data.append(dicom_files[i].ImagePositionPatient)
    coord_data = pd.DataFrame(scan_coordinate_data, columns=["x", "y", "z"])
    coord_data = coord_data[0:1015]
    # scan_coordinate_data = flatten_2d_array(scan_coordinate_data)
    # scan_coordinate_data = np.array(scan_coordinate_data)

    return training_scans, coord_data




# Mapping coordinate data from groundtruth mask/label to mri training data
def MappingCoordinateData(filename_label, coord_data):
    # Load in mesh of label data
    mesh = trimesh.load_mesh(('C:/Users/GGPC/OneDrive/Desktop/Part 4 Project/Part4Project/SegmentationMasks/{}.ply').format(filename_label))

    # Convert the mesh vertices to a DataFrame
    vertices = pd.DataFrame(mesh.vertices, columns=["x", "y", "z"])

    print('Height of Paitent in mm: ', np.abs(coord_data.iloc[-1][2] - coord_data.iloc[0][2]))
    print('Length of Paitent AOI (tibia) in mm: ', np.abs(vertices.iloc[-1][2] - vertices.iloc[0][2]))

    vertices['z'] = np.round(vertices['z'] * 2) / 2
    coord_data['z'] = np.round(coord_data['z'] * 2) / 2

    merged_df = pd.merge(coord_data, vertices, on='z')
    condensed_df = merged_df.groupby('z').mean().reset_index()

    mapping_dict = dict(zip(condensed_df['z'], ['AOI']*len(condensed_df)))

    coord_data['SegmentationRegionSlice'] = coord_data['z'].map(mapping_dict).fillna('Outside of AOI')

    slices_aoi_start = (coord_data.loc[coord_data['SegmentationRegionSlice'] == 'AOI'].index)[0]
    slices_aoi_end = (coord_data.loc[coord_data['SegmentationRegionSlice'] == 'AOI'].index)[-1]
    slice_aoi_range = (slices_aoi_end - slices_aoi_start + 1)
    print('AOI Slice Start: ', slices_aoi_start)
    print('AOI Slice End: ', slices_aoi_end)
    print('AOI Slice Range: ', slice_aoi_range)

    # CSV Format 
    if False:
        coord_data.to_csv('tibia_mri_coord.csv')

    return slices_aoi_start, slices_aoi_end, slice_aoi_range, coord_data




def VoxelisationMask(filename_label, slice_aoi_range):
    # Load in mesh of label data
    mesh = trimesh.load_mesh(('C:/Users/GGPC/OneDrive/Desktop/Part 4 Project/Part4Project/SegmentationMasks/{}.ply').format(filename_label))

    # Convert the mesh vertices to a DataFrame
    vertices = pd.DataFrame(mesh.vertices, columns=["x", "y", "z"])

    # Convert the mesh to a PyntCloud object
    cloud = PyntCloud(vertices)

    # Set the desired resolution
    desired_resolution = [slice_aoi_range, 512, 512]

    # Voxelize the mesh using the PyntCloud voxelization module
    voxelgrid_id = cloud.add_structure("voxelgrid", n_x=desired_resolution[0], n_y=desired_resolution[1], n_z=desired_resolution[2])
    voxel_grid = cloud.structures[voxelgrid_id].get_feature_vector().reshape(desired_resolution)

    # Transpose and swap axes to change the voxel grid orientation
    voxel_grid = np.transpose(voxel_grid, axes=(2, 0, 1))

    # Resize the voxel grid to match the desired dimensions
    voxel_grid = resize(voxel_grid, desired_resolution, anti_aliasing=False)

    voxel_grid = np.where(voxel_grid > 0, 1, 0)

    print('Mask Slices Normalized to MRI Scans Shape (Purely AOI): ', voxel_grid.shape)

    return voxel_grid




# \\files.auckland.ac.nz\research\resmed202100086-tws ----> Address for raw data
scans_path = 'D:/MRI - Tairawhiti'
# filename_labels = ['R_tibia_15A', 'R_tibia_16A', 'R_tibia_4A']
filename_labels = ['R_tibia_15A']
train_mask_tibia_labels, training_scans, start_slices_aoi, end_slices_aoi, slice_aoi_ranges  = [], [], [], [], []

for filename_label in filename_labels:
    print('\n')
    print(('{}'.format(filename_label)))

    training_scan, coord_data = TrainingMRIScans(scans_path)
    slices_aoi_start, slices_aoi_end, slice_aoi_range, coord_data = MappingCoordinateData(filename_label, coord_data)
    voxel_grid = VoxelisationMask(filename_label, slice_aoi_range)

    train_mask_tibia = np.zeros((1015, 512, 512))
    train_mask_tibia[(slices_aoi_start):(slices_aoi_end+1)] = voxel_grid
    # train_mask_tibia[(slices_aoi_start):(slices_aoi_end)] = voxel_grid
    train_mask_tibia_labels.append(train_mask_tibia)

    training_scans.append(training_scan)

    start_slices_aoi.append(slices_aoi_start)
    end_slices_aoi.append(slices_aoi_end)
    slice_aoi_ranges.append(slice_aoi_range)

    print('\n')

max_slice_aoi_range = np.max(slice_aoi_ranges)
min_start_slice_aoi = np.min(start_slices_aoi)
max_end_slice_aoi = np.max(end_slices_aoi)

for patient in range(len(train_mask_tibia_labels)):
    train_mask_tibia_labels[patient] = train_mask_tibia_labels[patient][min_start_slice_aoi:max_end_slice_aoi]
    training_scans[patient] = training_scans[patient][min_start_slice_aoi:max_end_slice_aoi]

train_mask_tibia_labels = np.array(train_mask_tibia_labels)
training_scans = np.array(training_scans)

# Determines image dataset size for UNet model
# training_scans_reshape = train_mask_tibia_labels.reshape((1, 10, 512, 512))
# train_mask_tibia_labels_reshape = training_scans.reshape((1, 10, 512, 512))
training_scans = training_scans[:, :10, :, :]
train_mask_tibia_labels = train_mask_tibia_labels[:, :1, :, :]

# Free up memory occupied by the original arrays
# del training_scans
# del train_mask_tibia_labels
# training_scans = training_scans_reshape
# train_mask_tibia_labels = train_mask_tibia_labels_reshape
# del training_scans_reshape
# del train_mask_tibia_labels_reshape


print('Number of Paitents: ', (training_scans.shape)[0])
print('Training Scans Input Shape: ', training_scans.shape)
print('Training Masks Input Shape: ', train_mask_tibia_labels.shape)
get_ram_usage(training_scans, 'training_scans')
get_ram_usage(train_mask_tibia_labels, 'train_mask_tibia_labels')



# Define U-Net model
def unet_model(input_shape):
    inputs = Input(input_shape)
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    # Bottleneck
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    # Decoder
    up6 = Conv2D(512, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)
    
    up7 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)
    
    up8 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)
    
    up9 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    
    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Dice Coefficient Loss Function 
def dice_coefficient(y_true, y_pred):
    smooth = 1e-5
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice


# Reformat image data structure
training_scans_reshaped = np.concatenate(training_scans, axis=0)
training_scans = training_scans_reshaped.reshape((-1, 512, 512, 1))
train_mask_tibia_labels_reshaped = np.concatenate(train_mask_tibia_labels, axis=0)
train_mask_tibia_labels = train_mask_tibia_labels_reshaped.reshape((-1, 512, 512, 1))

# Split the data into training and validation sets\
images_train, images_val, labels_train, labels_val = train_test_split(training_scans, train_mask_tibia_labels, test_size=0.2, random_state=0)
unseen_scan_model = np.array(training_scans[2][100])
images_train = images_train.astype('float32') / 255.0
images_val = images_val.astype('float32') / 255.0

print(images_train.shape)
print(labels_train.shape)
print(images_train.dtype)
print(labels_train.dtype)
print(images_val.shape)
print(labels_val.shape)
print(images_val.dtype)
print(labels_val.dtype)
print(unseen_scan_model.shape)

# Expand dimensions for the channel (grayscale) dimension
# images_train = np.expand_dims(images_train, axis=-1)
# images_val = np.expand_dims(images_val, axis=-1)
# labels_train = np.expand_dims(labels_train, axis=-1)
# labels_val = np.expand_dims(labels_val, axis=-1)

# Create an instance of the U-Net model
input_shape = (512, 512, 1)  # For grayscale images

# Create an instance of the U-Net model
model = unet_model(input_shape)

# Compile the model
# Binary Cross Entropy Loss Function
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Dice Coefficient Loss Function
# model.compile(optimizer=Adam(), loss=dice_coefficient, metrics=['accuracy'])

# Train the model
# Hyperparameter tuning -> batch_size
model.fit(x=images_train, y=labels_train, batch_size=32, epochs=1, validation_data=(images_val, labels_val))
# Evaluate the model
loss, accuracy = model.evaluate(x=images_val, y=labels_val)

# Perform inference on new, unseen MRI scans
predictions = model.predict(unseen_scan_model)