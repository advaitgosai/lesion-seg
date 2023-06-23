import cv2
import nibabel as nib
import numpy as np
import os
import warnings
import subprocess
import torchio as tio

def recreate():
    # Brain images
    input_nii_dir = 'input/'
    
    model_names = [f.name for f in os.scandir('predicted/2D/') if f.is_dir()]
    
    for model in model_names:
        # Results from model (should contain a directory having 128 2D mask slices for each corresponding brain image)
        slice_dir = f'predicted/2D/{model}/' 
        # Converted Mask images stored to raw predictions directory for preprocessing
        output_dir = f'predicted/3D/raw/{model}/'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for input_nii_filename in os.listdir(input_nii_dir):
            if input_nii_filename.endswith('.nii'):
                input_nii_path = os.path.join(input_nii_dir, input_nii_filename)
                output_nii_path = os.path.join(output_dir, os.path.splitext(input_nii_filename)[0] + '_pred.nii')
                nii_matrix = nib.load(input_nii_path)
                output_nii = nib.Nifti1Image(nii_matrix.get_fdata(), nii_matrix.affine, nii_matrix.header.copy())
                output_nii_matrix = output_nii.get_fdata()
                slice_subdir = os.path.join(slice_dir, input_nii_filename.replace('.nii', ''))
                num_images = len(os.listdir(slice_subdir))

                for i in range(num_images):
                    filename = os.path.join(slice_subdir, f"output-slice{i:03d}.jpg")
            
                    if os.path.isfile(filename):
                        slice_img = cv2.imread(filename)
                        slice_gray = cv2.cvtColor(slice_img, cv2.COLOR_BGR2GRAY)
                        rotated_slice_gray = cv2.rotate(slice_gray, cv2.ROTATE_90_CLOCKWISE)
                        output_nii_matrix[:, :, i] = rotated_slice_gray.astype(np.float32)
                    else:
                        warnings.warn(f'File "{filename}" does not exist.')

                # Flip the output Nifti image along the first axis if necessary
                input_signs = np.sign(nii_matrix.affine.diagonal())
                output_signs = np.sign(output_nii.affine.diagonal())
                if not np.allclose(input_signs, output_signs):
                    output_nii_matrix = np.flip(output_nii_matrix, axis=0)

                output_nii = nib.Nifti1Image(output_nii_matrix, output_nii.affine, output_nii.header)
                nib.save(output_nii, output_nii_path)

def transform():
    model_names = [f.name for f in os.scandir('predicted/3D/raw/') if f.is_dir()]

    for model in model_names:
        input_tr_dir = f'predicted/3D/raw/{model}'
        output_tr_dir = f'predicted/3D/mod/{model}'
        
        if not os.path.exists(output_tr_dir):
            os.makedirs(output_tr_dir)

        # Threshold
        input_files = os.listdir(input_tr_dir)
        for input_file in input_files:
            if input_file.endswith('.nii') or input_file.endswith('.nii.gz'):
                input_path = os.path.join(input_tr_dir, input_file)
                output_path = os.path.join(output_tr_dir, input_file)
                cmd = f"fslmaths {input_path} -thr 150 -bin {output_path}"
                subprocess.run(cmd, shell=True)

        # Crop
        for filename in os.listdir(output_tr_dir):
            if filename.endswith('.nii.gz'): # check if file is NIfTI image
                # load image and apply transform
                input_path = os.path.join(output_tr_dir, filename)
                output_path = os.path.join(output_tr_dir, filename)
                image = tio.ScalarImage(input_path)
                transform = tio.CropOrPad((182, 218, 182))
                output = transform(image)
                output.save(output_path)
