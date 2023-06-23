import os
import glob
import numpy as np
import nibabel as nib
import numba as nb

def ensemble():

    @nb.jit(nopython=True)
    def check_overlap(mask1, mask2, window_size, overlap_threshold):
        overlap = np.sum(np.logical_and(mask1 == 1, mask2 == 1))
        return (overlap / (window_size**3)) >= overlap_threshold

    @nb.jit(nopython=True)
    def loop_through_voxels(mask1_data, mask2_data, final_mask_data, x_dim, y_dim, z_dim, window_size, overlap_threshold):
        for x in range(x_dim - window_size + 1):
            for y in range(y_dim - window_size + 1):
                for z in range(z_dim - window_size + 1):
                    window_mask1 = mask1_data[x:x+window_size, y:y+window_size, z:z+window_size]
                    window_mask2 = mask2_data[x:x+window_size, y:y+window_size, z:z+window_size]

                    if check_overlap(window_mask1, window_mask2, window_size, overlap_threshold):
                        final_mask_data[x:x+window_size, y:y+window_size, z:z+window_size] = \
                            np.logical_or(window_mask1, window_mask2)
        return final_mask_data

    def process_masks(mask1_path, mask2_path, output_path, window_size, overlap_threshold):
        mask1_nii = nib.load(mask1_path)
        mask2_nii = nib.load(mask2_path)
        mask1_data = mask1_nii.get_fdata()
        mask2_data = mask2_nii.get_fdata()
        final_mask_data = np.zeros_like(mask1_data)
        x_dim, y_dim, z_dim = mask1_data.shape
        final_mask_data = loop_through_voxels(mask1_data, mask2_data, final_mask_data, x_dim, y_dim, z_dim, window_size, overlap_threshold)
        final_mask_nii = nib.Nifti1Image(final_mask_data, mask1_nii.affine)
        nib.save(final_mask_nii, output_path)

    def process_directory(mod1, mod2, output_dir, window_size, overlap_threshold):
        os.makedirs(output_dir, exist_ok=True)
        
        dir1 = f'predicted/3D/mod/{mod1}'
        dir2 = f'predicted/3D/mod/{mod2}'
        
        mask_files1 = glob.glob(os.path.join(dir1, '*.nii.gz'))
        mask_files2 = glob.glob(os.path.join(dir2, '*.nii.gz'))

        for mask_file1 in mask_files1:
            matching_file = next((f for f in mask_files2 if os.path.basename(f) == os.path.basename(mask_file1)), None)
            if matching_file:
                mask_id = matching_file.split('/')[4]
                output_path = os.path.join(output_dir, mask_id)
                process_masks(mask_file1, matching_file, output_path, window_size, overlap_threshold)
        
    
    ensemble_dir1 = 'predicted/3D/ensembled/AttUNet_VNet_3_0.5'
    ensemble_dir2 = 'predicted/3D/ensembled/FC_DenseNet_FT_UNet++_3_0.75'
    if not os.path.exists(ensemble_dir1):
            os.makedirs(ensemble_dir1)
    if not os.path.exists(ensemble_dir2):
            os.makedirs(ensemble_dir2)
    process_directory('AttUNet', 'VNet', ensemble_dir1, 3, 0.5)
    process_directory('FC_DenseNet_FT', 'UNet++', ensemble_dir2, 3, 0.75)