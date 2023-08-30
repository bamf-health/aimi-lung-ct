import yaml
import argparse
import os
from pathlib import Path
from converter_utils import DicomToNiiConverter, NiiToDicomConverter
from bamf_nnunet_inference import BAMFnnUNetInference
from lung_processor import LungPostProcessor
from io_utils import DotDict, get_path
import shutil


def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config


def run_nnunet(source_ct_dir, target_dir, output_nodules_seg_name, output_lesions_seg_name, num_folds=5, organ_label=9):
    """
    Convert list of dcm files to a single nii.gz file
    :param: source_ct_dir - dir containing list of dcm files
    :param: target_dir - dir to write segmented dcm masks too
    :param: num_folds - number of folds the nnUNet model was trained for
    :param: organ_label - label of lung segment in AIMI dataset    
    """    
    temp_nii_dir = "/tmp/nii-input"
    Path(temp_nii_dir).mkdir(parents=True, exist_ok=True)
    temp_ct_path = get_path(temp_nii_dir, "ct_0000.nii.gz")

    temp_folds_dir = "/tmp/folds"
    Path(temp_folds_dir).mkdir(parents=True, exist_ok=True)

    # Prepare config for nnUNet models for nodules and nsclc_rg
    WEIGHTS_FOLDER_NODULES = os.environ["WEIGHTS_FOLDER_NODULES"]
    WEIGHTS_FOLDER_NSCLC_RG = os.environ["WEIGHTS_FOLDER_NSCLC_RG"]
    TASK_NAME_NODULES = os.environ["TASK_NAME_NODULES"]
    TASK_NAME_NSCLC_RG = os.environ["TASK_NAME_NSCLC_RG"]
    model_path_nodules = get_path(WEIGHTS_FOLDER_NODULES, f"3d_fullres/{TASK_NAME_NODULES}/nnUNetTrainerV2__nnUNetPlansv2.1")
    model_path_nsclc_rg = get_path(WEIGHTS_FOLDER_NSCLC_RG, f"3d_fullres/{TASK_NAME_NSCLC_RG}/nnUNetTrainerV2__nnUNetPlansv2.1")
    nnunet_inference_model = BAMFnnUNetInference()

    # convert dcm to nii
    converter = DicomToNiiConverter()
    converter.dcm_to_nii(source_ct_dir, temp_ct_path)

    #################################################
    # Infer using nnUNet model across all folds     #
    # for Task777_CT_Nodules                        #
    #################################################
    organ_name_nodules_prefix = "ct_nodules_fold"
    for fold_idx in range(num_folds):
        organ_name_nodules_prefix_fold = f"{organ_name_nodules_prefix}_{fold_idx}"
        context = {
            'checkpoint_path': model_path_nodules,
            'input_file': temp_ct_path,
            'pt_file': None,
            'prediction_save': temp_folds_dir,
            'predict_aug': False,
            'softmax': False,
            'organ_name': organ_name_nodules_prefix_fold,
            'fold': fold_idx,
        }
        context = DotDict(context)
        print(f"inferring for fold {fold_idx} for task {TASK_NAME_NODULES}")
        print(context)
        # nnUNet creates below file internally. Format: temp_dir/ct_nodules_fold_0.nii.gz        
        output_seg_nii_file = f"{organ_name_nodules_prefix_fold}.nii.gz"
        output_seg_nii_path = get_path(temp_folds_dir, output_seg_nii_file)
        if not os.path.isfile(output_seg_nii_path):
            nnunet_inference_model.handle(context=context)

    #################################################
    # Infer using nnUNet model across all folds     #
    # for Task775_CT_NSCLC_RG                       #
    #################################################
    organ_name_nsclc_rg_prefix = "ct_nsclc_rg_fold"
    for fold_idx in range(num_folds):
        organ_name_nsclc_rg_prefix_fold = f"{organ_name_nsclc_rg_prefix}_{fold_idx}"
        context = {
            'checkpoint_path': model_path_nsclc_rg,
            'input_file': temp_ct_path,
            'pt_file': None,
            'prediction_save': temp_folds_dir,
            'predict_aug': False,
            'softmax': False,
            'organ_name': organ_name_nsclc_rg_prefix_fold,
            'fold': fold_idx,
        }
        context = DotDict(context)
        print(f"inferring for fold {fold_idx} for task {TASK_NAME_NSCLC_RG}")
        print(context)
        # nnUNet creates below file internally. Format: temp_dir/ct_nsclc_rg_fold_0.nii.gz
        output_seg_nii_file = f"{organ_name_nsclc_rg_prefix_fold}.nii.gz"
        output_seg_nii_path = get_path(temp_folds_dir, output_seg_nii_file)
        if not os.path.isfile(output_seg_nii_path):
            nnunet_inference_model.handle(context=context)

    # ensemble and post process
    lung_post_processor = LungPostProcessor()
    # out_file_path_nii_final = get_path(temp_folds_dir, output_seg_name)
    output_nodules_seg_path = get_path(temp_folds_dir, output_nodules_seg_name)
    output_lesions_seg_path = get_path(temp_folds_dir, output_lesions_seg_name)
    lung_post_processor.postprocessing(
        save_path=temp_folds_dir,
        ct_path=temp_ct_path,
        output_nodules_seg_path=output_nodules_seg_path,
        output_lesions_seg_path=output_lesions_seg_path,
        organ_name_nodules_prefix=organ_name_nodules_prefix,
        organ_name_nsclc_rg_prefix=organ_name_nsclc_rg_prefix,        
        lung_label=organ_label
        )

    #################################################
    # Convert Nifties back to dcm                   #
    #################################################
    dcmqi_package_path = os.environ["DCMQI_PACKAGE_PATH"]
    converter = NiiToDicomConverter(dcmqi_package_path)

    # convert nii output back to dcm for Task777_CT_Nodules
    # Example output_nodules_seg_name: "seg_nodules_ensemble.nii.gz" 
    output_nodules_seg_name_dcm = output_nodules_seg_name.split('.')[0].strip() + ".dcm"
    target_segmented_dcm_file = get_path(target_dir, output_nodules_seg_name_dcm)
    nodules_success = converter.convert_nii_to_dcm(
        nii_path=Path(output_nodules_seg_path),
        dcm_ref_dir=Path(source_ct_dir),
        dcm_out_file=Path(target_segmented_dcm_file),
        dicom_seg_meta_json=Path("dicom_seg_meta.json"),
        add_background_label=False
    )
    # Safety check: If dicom conversion fails, ship the nii file
    if not nodules_success:
        target_segmented_nii_file = get_path(target_dir, output_nodules_seg_name)
        shutil.copyfile(output_nodules_seg_path, target_segmented_nii_file)
    print("Execution of nodules segmentation complete!")

    # convert nii output back to dcm for Task775_CT_NSCLC_RG
    # Example output_lesions_seg_name: "seg_lesions_ensemble.nii.gz"     
    output_lesions_seg_name_dcm = output_lesions_seg_name.split('.')[0].strip() + ".dcm"
    target_segmented_dcm_file = get_path(target_dir, output_lesions_seg_name_dcm)
    lesions_success = converter.convert_nii_to_dcm(
        nii_path=Path(output_lesions_seg_path),
        dcm_ref_dir=Path(source_ct_dir),
        dcm_out_file=Path(target_segmented_dcm_file),
        dicom_seg_meta_json=Path("dicom_seg_meta.json"),
        add_background_label=False
    )
    # Safety check: If dicom conversion fails, ship the nii file
    if not lesions_success:
        target_segmented_nii_file = get_path(target_dir, output_lesions_seg_name)
        shutil.copyfile(output_lesions_seg_path, target_segmented_nii_file)
    print("Execution of lesions segmentation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and display YAML configuration")
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()
    config_path = args.config
    config = load_config(config_path)

    # Load arguments from config
    general = config.get("general", {})
    data_base_dir = general.get("data_base_dir")
    modules = config.get("modules", {})
    nnunet_runner = modules.get("NNUnetRunner", {})
    source_ct_dir = nnunet_runner.get("source_ct_dir")
    source_ct_dir = os.path.join(data_base_dir, source_ct_dir)
    target_dir = nnunet_runner.get("target_dir")
    target_dir = os.path.join(data_base_dir, target_dir)
    num_folds = int(nnunet_runner.get("num_folds"))
    organ_label = int(nnunet_runner.get("organ_label"))
    output_nodules_seg_name = nnunet_runner.get("output_nodules_seg_name")
    output_lesions_seg_name = nnunet_runner.get("output_lesions_seg_name")

    # Run the model
    run_nnunet(
        source_ct_dir=source_ct_dir,
        target_dir=target_dir,
        output_nodules_seg_name=output_nodules_seg_name,
        output_lesions_seg_name=output_lesions_seg_name,
        num_folds=5,
        organ_label=organ_label
        )
