import nibabel as nib
import numpy as np
import os
import pandas as pd

def calculate():
    CST_right_mask_nii = nib.load('assets/thr3Right12Subj_sum_bin_warped_thr50_extCST_fdt_paths_mul_BrainMask_noCerebellum.nii')
    CST_left_mask_nii = nib.load('assets/thr3Left12Subj_sum_bin_warped_thr50_extCST_fdt_paths_mul_BrainMask_noCerebellum.nii')

    mask_dir = 'output/'
    mask_names = [f for f in os.listdir(mask_dir)]
    mask_names.sort()

    lesion_load_left_probabilistic = np.zeros(len(mask_names))
    lesion_load_left_geometric = np.zeros(len(mask_names))
    lesion_load_right_probabilistic = np.zeros(len(mask_names))
    lesion_load_right_geometric = np.zeros(len(mask_names))
    lesion_volume = np.zeros(len(mask_names))

    def compute_geom_prob_load(mask_dir, mask_names, CST_left_mask_nii, CST_right_mask_nii):

        for i in range(len(mask_names)):
            brain_mask_nii = nib.load(os.path.join(mask_dir, mask_names[i]))
            lesion_volume[i] = np.sum(brain_mask_nii.get_fdata())
            lesion_load_left_probabilistic[i] = np.sum((CST_left_mask_nii.get_fdata() * brain_mask_nii.get_fdata()) / 12)
            lesion_load_right_probabilistic[i] = np.sum((CST_right_mask_nii.get_fdata() * brain_mask_nii.get_fdata()) / 12)

            # Geometric lesion load
            numerator_l = (CST_left_mask_nii.get_fdata() * brain_mask_nii.get_fdata()) > 0
            denominator_l = CST_left_mask_nii.get_fdata() > 0
            numerator_r = (CST_right_mask_nii.get_fdata() * brain_mask_nii.get_fdata()) > 0
            denominator_r = CST_right_mask_nii.get_fdata() > 0
            slice_LL_left_geometric = 0
            slice_LL_right_geometric = 0
            for j in range(182):
                if np.sum(numerator_l[:, :, j]):
                    slice_LL_left_geometric = np.sum(numerator_l[:, :, j]) / np.sum(denominator_l[:, :, j])
                    lesion_load_left_geometric[i] += slice_LL_left_geometric
                if np.sum(numerator_r[:, :, j]):
                    slice_LL_right_geometric = np.sum(numerator_r[:, :, j]) / np.sum(denominator_r[:, :, j])
                    lesion_load_right_geometric[i] += slice_LL_right_geometric

        return (lesion_load_left_probabilistic, lesion_load_right_probabilistic,
                lesion_load_left_geometric, lesion_load_right_geometric, lesion_volume)

    (lesion_load_left_probabilistic, lesion_load_right_probabilistic,
    lesion_load_left_geometric, lesion_load_right_geometric, lesion_volume) = compute_geom_prob_load(
        mask_dir, mask_names, CST_left_mask_nii, CST_right_mask_nii)

    f_xI_left = np.zeros((182,218,182))
    f_xI_right = np.zeros((182,218,182))

    lesion_load_left_combined_new = np.zeros((0, 1))
    lesion_load_right_combined_new = np.zeros((0, 1))

    MaxIntensityonSlicesLeft = np.zeros((13, 1))
    MaxIntensityonSlicesRight = np.zeros((13, 1))

    lesion_load_left_combined_new_vals = np.zeros(len(mask_names))
    lesion_load_right_combined_new_vals = np.zeros(len(mask_names))

    def final_cal_max_intensity(lesionmask_weighted_withCSTof12Subj_left,lesionmask_weighted_withCSTof12Subj_right):
        for j in range(182):
            for k in range(1,13):
                slicemask_UniqueVoxelInt_left = np.unique(lesionmask_weighted_withCSTof12Subj_left[:, :, j] > k)
                VoxInt_Cnt_l = slicemask_UniqueVoxelInt_left.size
                if VoxInt_Cnt_l > MaxIntensityonSlicesLeft[k, 0]:

                    MaxIntensityonSlicesLeft[k, 0] = VoxInt_Cnt_l
                
            for k in range(1,13):
                slicemask_UniqueVoxelInt_right = np.unique(lesionmask_weighted_withCSTof12Subj_right[:, :, j] > k)
                VoxInt_Cnt_r = slicemask_UniqueVoxelInt_right.size
                if VoxInt_Cnt_r > MaxIntensityonSlicesRight[k, 0]: 

                    MaxIntensityonSlicesRight[k, 0] = VoxInt_Cnt_r

    ##left calculation            
        for j in range(182):
            slicemask_UniqueVoxelInt_left = np.unique(lesionmask_weighted_withCSTof12Subj_left[:, :, j])
            VoxInt_Cnt_l = slicemask_UniqueVoxelInt_left.shape[0]
            slicemask_UniqueVoxelCount_left = np.zeros((VoxInt_Cnt_l - 1, 1))
            slice_cst_UniqueVoxelCount_left = np.zeros((VoxInt_Cnt_l - 1, 1))
            overlap_voxInd_l = np.nonzero(lesionmask_weighted_withCSTof12Subj_left[:, :, j])

            if VoxInt_Cnt_l > 1:
                for k in range(1, VoxInt_Cnt_l):
                    slicemask_UniqueVoxelCount_left[k - 1, 0] = np.sum(lesionmask_weighted_withCSTof12Subj_left[:, :, j] >= slicemask_UniqueVoxelInt_left[k])
                    slice_cst_UniqueVoxelCount_left[k - 1, 0] = MaxIntensityonSlicesLeft[int(slicemask_UniqueVoxelInt_left[k]),0]

                    slice_Ratio_f_xI_l = np.divide(slicemask_UniqueVoxelCount_left, slice_cst_UniqueVoxelCount_left)
            
                for k in range(181):
                    for l in range(218):
                        if lesionmask_weighted_withCSTof12Subj_left[k, l, j]:
                            voxint_uniqueIdx = np.argwhere(slicemask_UniqueVoxelInt_left == lesionmask_weighted_withCSTof12Subj_left[k, l, j])[0, 0]
                            f_xI_left[k, l, j] = slice_Ratio_f_xI_l[voxint_uniqueIdx - 1]
        lesion_load_left_combined_new = np.sum(1/12 * lesionmask_weighted_withCSTof12Subj_left * f_xI_left)

    #right calculation
        for j in range(182):
            slicemask_UniqueVoxelInt_right = np.unique(lesionmask_weighted_withCSTof12Subj_right[:, :, j])
            VoxInt_Cnt_r = slicemask_UniqueVoxelInt_right.shape[0]
            slicemask_UniqueVoxelCount_right = np.zeros((VoxInt_Cnt_r - 1, 1))
            slice_cst_UniqueVoxelCount_right = np.zeros((VoxInt_Cnt_r - 1, 1))
        
            if VoxInt_Cnt_r > 1:
                for k in range(1,VoxInt_Cnt_r):
                    slicemask_UniqueVoxelCount_right[k - 1, 0] = np.sum(np.sum(lesionmask_weighted_withCSTof12Subj_right[:, :, j] >= slicemask_UniqueVoxelInt_right[k]))
                    slice_cst_UniqueVoxelCount_right[k - 1, 0] = MaxIntensityonSlicesRight[int(slicemask_UniqueVoxelInt_right[k]),0]

                    slice_Ratio_f_xI_r = slicemask_UniqueVoxelCount_right / slice_cst_UniqueVoxelCount_right
                for k in range(181):
                    for l in range(218):
                        if lesionmask_weighted_withCSTof12Subj_right[k, l, j]:
                            voxint_uniqueIdx = np.argwhere(slicemask_UniqueVoxelInt_right == lesionmask_weighted_withCSTof12Subj_right[k, l, j])[0][0]
                            f_xI_right[k, l, j] = slice_Ratio_f_xI_r[voxint_uniqueIdx - 1]
                    

        lesion_load_right_combined_new = np.sum(np.sum(np.sum(1/12 * lesionmask_weighted_withCSTof12Subj_right * f_xI_right)))

        return (lesion_load_right_combined_new,lesion_load_left_combined_new)

    for i in range(len(mask_names)):
        brain_mask_nii = nib.load(os.path.join(mask_dir, mask_names[i]))
        lesionmask_weighted_withCSTof12Subj_left = CST_left_mask_nii.get_fdata().astype(np.float32) * brain_mask_nii.get_fdata().astype(np.float32)
        CST_left_mask_matrix = CST_left_mask_nii.get_fdata().astype(np.float32)
        lesionmask_weighted_withCSTof12Subj_right = CST_right_mask_nii.get_fdata().astype(np.float32) * brain_mask_nii.get_fdata().astype(np.float32)
        CST_right_mask_matrix = CST_right_mask_nii.get_fdata().astype(np.float32)
        lesion_load_right_combined_new_vals[i], lesion_load_left_combined_new_vals[i] = final_cal_max_intensity(lesionmask_weighted_withCSTof12Subj_left, lesionmask_weighted_withCSTof12Subj_right)

    df_final = pd.DataFrame()
    df_final['Mask_ID'] = [m.split('.')[0] for m in mask_names]
    df_final['Lesion_Volume_Pred'] = lesion_volume
    df_final['Lesion_Load_Left_Pred'] = lesion_load_left_combined_new_vals
    df_final['Lesion_Load_Right_Pred'] = lesion_load_right_combined_new_vals
    df_final.round(3).to_csv(mask_dir + 'lesion_calculations_all.csv', index=False)
