import os
import torchio as tio
import subprocess

def resample():

    # set input and output folder paths
    input_dir = 'input'
    output_dir = 'preprocessed/3D'

    # create output folder if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # iterate over all files in the input folder
    for filename in os.listdir(input_dir):
        if filename.endswith('.nii') or filename.endswith('nii.gz'): # check if file is NIfTI image
            # load image and apply transform
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            image = tio.ScalarImage(input_path)
            transform = tio.CropOrPad((256, 256, 128))
            output = transform(image)
            # save transformed image to output folder
            output.save(output_path)

def slice():
    # Set the path to your input directory containing the NII files
    input_dir = "preprocessed/3D"

    # Set the path to your output directory where the converted images will be saved
    output_dir = "preprocessed/2D"

    # create output folder if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through each file in the input directory
    for file_name in os.listdir(input_dir):
        # Check if the file is a NII file
        if file_name.endswith(".nii"):
            # Set the output folder name to be the same as the file name (without the extension)
            output_folder_name = os.path.splitext(file_name)[0]
            # Create the output folder if it doesn't already exist
            output_folder_path = os.path.join(output_dir, output_folder_name)
            os.makedirs(output_folder_path, exist_ok=True)
            # Use med2image to convert the NII file to PNG format
            input_file_path = os.path.join(input_dir, file_name)
            subprocess.run(["med2image", "-i", input_file_path, "-d", output_folder_path])
