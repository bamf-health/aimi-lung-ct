import os
import SimpleITK as sitk
import numpy as np
from skimage import measure
from io_utils import get_path


class LungPostProcessor:
    def __init__(self):
        pass

    def get_ensemble(self, save_path, organ_name_prefix, label, num_folds=5, th=0.6):
        """
        Perform ensemble segmentation on medical image data.

        Args:
            save_path (str): Path to the directory where segmentation results will be saved.
            organ_name_prefix (str): A prefix to construct and fetch segmented paths that nnUNet creates
            label (int): Label value for the segmentation.
            num_folds (int, optional): Number of folds for ensemble. Default is 5.
            th (float, optional): Threshold value. Default is 0.6.

        Returns:
            np.ndarray: Segmentation results.
        """
        for fold in range(num_folds):
            inp_seg_file = f"{organ_name_prefix}_{fold}.nii.gz"
            inp_seg_path = get_path(save_path, inp_seg_file)
            seg_data = sitk.GetArrayFromImage(sitk.ReadImage(inp_seg_path))
            seg_data[seg_data != label] = 0
            seg_data[seg_data == label] = 1
            if fold == 0:
                segs = np.zeros(seg_data.shape)
            segs += seg_data
        segs = segs / 5
        segs[segs < th] = 0
        segs[segs >= th] = 1
        return segs

    def n_connected(self, img_data):
        """
        Get the largest connected component in a binary image.

        Args:
            img_data (np.ndarray): image data.

        Returns:
            np.ndarray: Processed image with the largest connected component.
        """
        img_filtered = np.zeros(img_data.shape)
        blobs_labels = measure.label(img_data, background=0)
        lbl, counts = np.unique(blobs_labels, return_counts=True)
        lbl_dict = {}
        for i, j in zip(lbl, counts):
            lbl_dict[i] = j
        sorted_dict = dict(sorted(lbl_dict.items(), key=lambda x: x[1], reverse=True))
        count = 0

        for key, value in sorted_dict.items():
            if count >= 1 and count <= 2 and value > 20:
                print(key, value)
                img_filtered[blobs_labels == key] = 1
            count += 1

        img_data[img_filtered != 1] = 0
        return img_data
    
    def get_lungs(self, save_path, lesion_prefix, th=0.6):
        for fold_idx in range(5):
            nnunet_seg_file_fold = f"{lesion_prefix}_{fold_idx}.nii.gz"
            ip_path = get_path(save_path, nnunet_seg_file_fold)
            seg_data = sitk.GetArrayFromImage(sitk.ReadImage(ip_path))
            seg_data[seg_data > 0] = 1
            if fold_idx == 0:
                segs = np.zeros(seg_data.shape)
            segs += seg_data
        segs = segs / 5
        segs[segs < th] = 0
        segs[segs >= th] = 1
        return segs

    def get_seg_img(self, lungs, nodules, ct_path):
        seg_data = np.zeros(lungs.shape)
        seg_data[lungs == 1] = 1
        seg_data[nodules == 1] = 2
        ref = sitk.ReadImage(ct_path)
        seg_img = sitk.GetImageFromArray(seg_data)
        seg_img.CopyInformation(ref)
        return seg_img
    
    def postprocessing(
            self,
            save_path: str,
            ct_path: str,
            output_nodules_seg_path: str,
            output_lesions_seg_path: str,            
            organ_name_nodules_prefix: str,
            organ_name_nsclc_rg_prefix: str,            
            lung_label: int
            ):
        """
        Perform postprocessing and writes simpleITK Image

        Args:
            save_path (str): Path to save inference results.
            ct_path (str): Path to input CT image.
            output_nodules_seg_path (str): Path to write final nodules segment mask to
            output_lesions_seg_path (str): Path to write final lesions segment mask to            
            organ_name_nodules_prefix (str): base name of the output mask from nnUNet for Task777_CT_Nodules
            organ_name_nodules_prefix (str): base name of the output mask from nnUNet for Task775_CT_NSCLC_RG
            lung_label (str): label of lung assigned in AIMI dataset
        Returns:
            None
        """
        nodules_seg_absent = not os.path.isfile(output_nodules_seg_path)
        lesions_seg_absent = not os.path.isfile(output_lesions_seg_path)
        if nodules_seg_absent or lesions_seg_absent:
            lungs = self.get_lungs(save_path, lesion_prefix=organ_name_nsclc_rg_prefix)
            lungs = self.n_connected(lungs)
            nodules = self.get_ensemble(
                save_path=save_path,
                organ_name_prefix=organ_name_nodules_prefix,
                label=lung_label,
                num_folds=5,
                th=0.6
                )
            lesions = self.get_ensemble(
                save_path=save_path,
                organ_name_prefix=organ_name_nsclc_rg_prefix,
                label=lung_label,
                num_folds=5,
                th=0.6
                )
            nodules[lungs == 0] = 0
            lesions[lungs == 0] = 0
            nodules_seg_img = self.get_seg_img(np.copy(lungs), np.copy(nodules), ct_path)
            lesions_seg_img = self.get_seg_img(np.copy(lungs), np.copy(lesions), ct_path)
            sitk.WriteImage(nodules_seg_img, output_nodules_seg_path)
            sitk.WriteImage(lesions_seg_img, output_lesions_seg_path)

