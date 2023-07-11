# Exporting Training Scan and Mask Data
import os 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def WriteOutTextFile(data_array, save_dir, starting_slice):
    num_files = data_array.shape[0]
    
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for i in range(num_files):
        file_name = f"data_{i + starting_slice}.txt"
        file_path = save_dir + '/' + file_name
        np.savetxt(file_path, data_array[i], delimiter=',', fmt='%.4f')
        print(f"Saved {file_path}")


def WriteOutImagePNGFiles(data_array, save_dir, starting_slice):
    num_files = data_array.shape[0]

    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(num_files):
        file_name = f"image_{i + starting_slice}.png"
        file_path = save_dir + '/' + file_name
            
        # Scale the pixel values to the range [0, 255] if necessary
        image_data = ((data_array[i] * 255).astype(np.uint8)).squeeze()
        # Create an Image object and save as PNG
        image = Image.fromarray(image_data)

        if True:
            image.save(file_path)
                
        print(f"Saved {file_path}")


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