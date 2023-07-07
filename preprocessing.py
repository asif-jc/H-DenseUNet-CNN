#Libraries
import numpy as np
import os
import pydicom
import SimpleITK as sitk
import pandas as pd
import trimesh
from pyntcloud import PyntCloud
from skimage.transform import resize
import sys

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


def ReadIn_MRIScans_Masks(scans_path):

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
 

    # Coordinate Data
    for paitent in folders:
        single_paitent_scans_path =  scans_path + '/{}'.format(paitent)
        for i in range (len(dicom_files)):
            scan_coordinate_data.append(dicom_files[i].ImagePositionPatient)

    #TEMP
    training_scans = training_scans[0:1015]
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



def preprocessing(scans_path, filename_labels):
    train_mask_tibia_labels, training_scans, start_slices_aoi, end_slices_aoi, slice_aoi_ranges  = [], [], [], [], []

    for filename_label in filename_labels:
        # print('\n')
        print(('{}'.format(filename_label)))

        training_scan, coord_data = ReadIn_MRIScans_Masks(scans_path)
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
    # training_scans_reshaped_array = np.squeeze(training_scans)
    # training_scans_reshaped_array = np.transpose(training_scans_reshaped_array, (1, 2, 3, 0))
    # training_scans_input = training_scans_reshaped_array[..., np.newaxis]
    training_scans_input_converted_array = np.squeeze(training_scans, axis=0)
    training_scans_input = np.expand_dims(training_scans_input_converted_array, axis=-1)

    # train_mask_tibia_labels_reshaped_array = np.squeeze(train_mask_tibia_labels)
    # train_mask_tibia_labels_reshaped_array = np.transpose(train_mask_tibia_labels_reshaped_array, (1, 2, 3, 0))
    # train_mask_tibia_labels_input = train_mask_tibia_labels_reshaped_array[..., np.newaxis]
    train_mask_tibia_labels_input_converted_array = np.squeeze(train_mask_tibia_labels, axis=0)
    train_mask_tibia_labels_input = np.expand_dims(train_mask_tibia_labels_input_converted_array, axis=-1)

    # training_scans = training_scans[:, :4, :, :]
    # train_mask_tibia_labels = train_mask_tibia_labels[:, :4, :, :]

    print('Number of Paitents: ', (training_scans.shape)[0])
    print('Training Scans Input Shape: ', training_scans_input.shape)
    print('Training Masks Input Shape: ', train_mask_tibia_labels_input.shape)
    get_ram_usage(training_scans_input, 'training_scans')
    get_ram_usage(train_mask_tibia_labels_input, 'train_mask_tibia_labels')

    return training_scans_input, train_mask_tibia_labels_input


# scans_path = 'D:/MRI - Tairawhiti'
# # filename_labels = ['R_tibia_15A', 'R_tibia_16A', 'R_tibia_4A']
# filename_labels = ['R_tibia_15A']
# train_mask_tibia_labels, training_scans, start_slices_aoi, end_slices_aoi, slice_aoi_ranges  = [], [], [], [], []

# for filename_label in filename_labels:
#     # print('\n')
#     print(('{}'.format(filename_label)))

#     training_scan, coord_data = ReadIn_MRIScans_Masks(scans_path)
#     slices_aoi_start, slices_aoi_end, slice_aoi_range, coord_data = MappingCoordinateData(filename_label, coord_data)
#     voxel_grid = VoxelisationMask(filename_label, slice_aoi_range)

#     train_mask_tibia = np.zeros((1015, 512, 512))
#     train_mask_tibia[(slices_aoi_start):(slices_aoi_end+1)] = voxel_grid
#     # train_mask_tibia[(slices_aoi_start):(slices_aoi_end)] = voxel_grid
#     train_mask_tibia_labels.append(train_mask_tibia)

#     training_scans.append(training_scan)

#     start_slices_aoi.append(slices_aoi_start)
#     end_slices_aoi.append(slices_aoi_end)
#     slice_aoi_ranges.append(slice_aoi_range)

#     print('\n')

# max_slice_aoi_range = np.max(slice_aoi_ranges)
# min_start_slice_aoi = np.min(start_slices_aoi)
# max_end_slice_aoi = np.max(end_slices_aoi)

# for patient in range(len(train_mask_tibia_labels)):
#     train_mask_tibia_labels[patient] = train_mask_tibia_labels[patient][min_start_slice_aoi:max_end_slice_aoi]
#     training_scans[patient] = training_scans[patient][min_start_slice_aoi:max_end_slice_aoi]

# train_mask_tibia_labels = np.array(train_mask_tibia_labels)
# training_scans = np.array(training_scans)

# # Determines image dataset size for UNet model
# # training_scans_reshaped_array = np.squeeze(training_scans)
# # training_scans_reshaped_array = np.transpose(training_scans_reshaped_array, (1, 2, 3, 0))
# # training_scans_input = training_scans_reshaped_array[..., np.newaxis]
# training_scans_input_converted_array = np.squeeze(training_scans, axis=0)
# training_scans_input = np.expand_dims(training_scans_input_converted_array, axis=-1)

# # train_mask_tibia_labels_reshaped_array = np.squeeze(train_mask_tibia_labels)
# # train_mask_tibia_labels_reshaped_array = np.transpose(train_mask_tibia_labels_reshaped_array, (1, 2, 3, 0))
# # train_mask_tibia_labels_input = train_mask_tibia_labels_reshaped_array[..., np.newaxis]
# train_mask_tibia_labels_input_converted_array = np.squeeze(train_mask_tibia_labels, axis=0)
# train_mask_tibia_labels_input = np.expand_dims(train_mask_tibia_labels_input_converted_array, axis=-1)

# # training_scans = training_scans[:, :4, :, :]
# # train_mask_tibia_labels = train_mask_tibia_labels[:, :4, :, :]

# print('Number of Paitents: ', (training_scans.shape)[0])
# print('Training Scans Input Shape: ', training_scans_input.shape)
# print('Training Masks Input Shape: ', train_mask_tibia_labels_input.shape)
# get_ram_usage(training_scans_input, 'training_scans')
# get_ram_usage(train_mask_tibia_labels_input, 'train_mask_tibia_labels')