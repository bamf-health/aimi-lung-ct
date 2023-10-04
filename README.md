# CT Lung Nodule Segmentation

This container combines the output of two models.

_Nodule segmentation model_

This model segments lung nodules (3-30mm diameter) from ct scans. This model was trained on the [DICOM-LIDC-IDRI-Nodules](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254) dataset.

_Lung segmentation model_
The lung segmentation model was trained on 411 and 111 lung CT scans from [NSCLC Radiomics](https://wiki.cancerimagingarchive.net/display/Public/NSCLC-Radiomics) and [NSCLC Radiogenomics](https://wiki.cancerimagingarchive.net/display/Public/NSCLC+Radiogenomics) respectively.

Both models were trained using [nnU-Net](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) framework.
The outputs of these two models are combined to produce the final output, with nodules taking precedence over lung segmentation.

The [model_performance](model_performance.ipynb) notebook contains the code to evaluate the model performance on the following IDC collections against a validation evaluated by a radiologist and a non-expert.

- [TCGA-LUAD](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=6881474)
- [TCGA-LUSC](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=16056484)
- [Lung PET-CT Dx](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70224216)
- [Anti-PD-1-Lung](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=41517500)
- [RIDER-Lung-PET-CT](https://wiki.cancerimagingarchive.net/display/Public/RIDER+Lung+PET-CT)
- [NSCLC Radiogenomics](https://wiki.cancerimagingarchive.net/display/Public/NSCLC+Radiogenomics)

### Running instructions

- Create an `input_dir_ct` containing list of `dcm` files corresponding to a given series_id for CT modality
- Create an `output_dir` to store the output from the model. This is a shared directory mounted on container at run-time. Please assign write permissions to this dir for container to be able to write data
- Next, pull the image from dockerhub:

  - `docker pull bamfhealth/bamf_nnunet_ct_lung:latest`

- Finally, let's run the container:
  - `docker run --gpus all -v {input_dir_ct}:/app/data/input_data/ct:ro -v {output_dir}:/app/data/output_data bamfhealth/bamf_nnunet_ct_lung:latest`
- Once the job is finished, the output inference mask(s) would be available in the `{output_dir}` folder
- Expected output from after successful container run is:
  - If nifty to dcm conversion is a success:
    - {output_dir}/seg_nodules_ensemble.dcm
    - {output_dir}/seg_lesions_ensemble.dcm
  - Else:
    - {output_dir}/seg_nodules_ensemble.nii.gz
    - {output_dir}/seg_lesions_ensemble.nii.gz
