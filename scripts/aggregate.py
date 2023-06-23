import nibabel as nib
import numpy as np
import os
import shutil

def aggregate():

    def voxel_dict_generator(nii_file):
        data = nii_file.get_fdata()
        intensities, counts = np.unique(data, return_counts=True)
        for i in range(len(intensities)):
            # Get the voxel locations where the intensity is equal to the current intensity
            locs = np.argwhere(data == intensities[i])
            # Convert the locations to a NumPy array
            locs_array = np.array(locs, dtype=np.int64)
            # Yield a tuple with the intensity as the key and the locations as the value
            yield (intensities[i], locs_array)

    nii_file = nib.load('assets/new-regions.nii.gz')
    voxel_dict = dict(voxel_dict_generator(nii_file))

    def find_mask(mask_id, folder_path):
        for filename in os.listdir(folder_path):
            if filename.startswith(mask_id):
                return filename

    regions_dict = {
        1.0: "predicted/3D/ensembled/FC_DenseNet_FT_UNet++_3_0.75/", 
        2.0: "predicted/3D/ensembled/AttUNet_VNet_3_0.5/", 
        3.0: "predicted/3D/mod/UNet++/", 
        4.0: "predicted/3D/ensembled/FC_DenseNet_FT_UNet++_3_0.75/"
    }

    ids = os.listdir('input/')
    ids_trunc = [i.split(".")[0] for i in ids if i.endswith('.nii')]
    input_mask_ids = np.sort(np.array(ids_trunc))

    for mask_id in input_mask_ids:
        print("##################### Mask ID:" , mask_id)
        all_regions_output = []
        for region in regions_dict:
            folder_paths = [m for m in regions_dict[region].split(',')]
            file_paths = []
            for f in folder_paths:
                file_paths.append(f + "/" + find_mask(mask_id, f))
            data_list = []
            for file_path in file_paths:
                nii_file = nib.load(file_path)
                data = nii_file.get_fdata()
                data_list.append(data)
            specific_voxels = voxel_dict[region]
            output_data = np.full(data_list[0].shape, -0.0)
            if region == 1.0 or region == 2.0 or region == 3.0 or region == 4.0:
                intensities_list = [data[tuple(specific_voxels.T)] for data in data_list]
                output_data[tuple(specific_voxels.T)] = np.prod(intensities_list, axis=0)
            else:
                output_data[tuple(specific_voxels.T)] = np.zeros_like(output_data[tuple(specific_voxels.T)])
            all_regions_output.append(output_data)
    
        result_mask = np.sum(all_regions_output, axis = 0)
        
        input_img = nib.load('input/' + mask_id + '.nii')
        affine = input_img.affine
        voxel_size = (1.0, 1.0, 1.0)
        nii_file = nib.Nifti1Image(result_mask, affine, nib.Nifti1Header())
        nii_file.set_qform(affine)
        nii_file.header['pixdim'][1:4] = voxel_size
        nii_file.header['xyzt_units'] = 2
        
        final_dir = 'output'
        if not os.path.exists(final_dir):
            os.makedirs(final_dir)
        nib.save(nii_file, os.path.join(final_dir, mask_id + '_final_prediction.nii.gz'))