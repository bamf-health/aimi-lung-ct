from __future__ import division
import argparse
import os
from pathlib import Path
from timeit import default_timer as timer
import numpy as np
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from nnunet.training.model_restore import load_model_and_checkpoint_files
import nrrd
import SimpleITK as sitk
import json
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class BAMFnnUNetInference:
    def initialize(self, context):
        self.trainer, self.params = load_model_and_checkpoint_files(
            context.checkpoint_path,
            context.fold,
            checkpoint_name="model_final_checkpoint",
        )
        self.trainer.initialize_network()
        self.trainer.network.load_state_dict(self.params[0]["state_dict"])
        self.mirror_axes = self.trainer.data_aug_params["mirror_axes"]
        self.context = context

    def preprocess(self):
        if self.context.pt_file:
            data = [str(self.context.input_file), str(self.context.pt_file)]
        else:
            data = [str(self.context.input_file)]
        print(data)
        data, s, self.properties = self.trainer.preprocess_patient(data)
        return data

    def inference(self, data):
        results = self.trainer.predict_preprocessed_data_return_seg_and_softmax(
            data, do_mirroring=self.context.predict_aug, mirror_axes=self.mirror_axes
        )[1]
        return results

    def convert_nifti_to_nrrd(self, labels: str = "labels.json"):
        """
        labels : path to user defined json file with segment names and other metadata
        """
        # print(self.output_file)
        img = sitk.ReadImage(self.output_file)
        op_path = self.output_file.replace(".nii.gz", ".seg.nrrd")
        sitk.WriteImage(img, op_path, useCompression=True, compressionLevel=9)
        data, h = nrrd.read(op_path)
        with open(os.path.join(self.context.checkpoint_path, labels), "rb") as f:
            cfm = json.load(f)
        h.update(cfm)
        nrrd.write(op_path, data, h)

    def postprocess(self, data):
        # can change the ouput path other than model dir
        self.output_dir = os.path.join(os.path.join(self.context.prediction_save))
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        self.output_file = os.path.join(
            self.output_dir, self.context.organ_name + ".nii.gz"
        )
        # optional
        softmax_ouput_file = os.path.join(self.output_dir, "temp_softmax")

        if "segmentation_export_params" in self.trainer.plans.keys():
            force_separate_z = self.trainer.plans["segmentation_export_params"][
                "force_separate_z"
            ]
            interpolation_order = self.trainer.plans["segmentation_export_params"][
                "interpolation_order"
            ]
            interpolation_order_z = self.trainer.plans["segmentation_export_params"][
                "interpolation_order_z"
            ]
        else:
            force_separate_z = None
            interpolation_order = 1
            interpolation_order_z = 0
        pred = data.transpose([0] + [i + 1 for i in self.trainer.transpose_backward])
        save_segmentation_nifti_from_softmax(
            pred,
            self.output_file,
            self.properties,
            interpolation_order,
            self.trainer.regions_class_order,
            None,
            None,
            softmax_ouput_file,
            None,
            force_separate_z=force_separate_z,
            interpolation_order_z=interpolation_order_z,
        )

        results = pred.tolist()

        return results

    def handle(self, context):
        """Entry point for default handler. It takes the data from the input request and returns
           the predicted outcome for the input.
        Args:
            context (Context): It is a JSON Object containing information pertaining to
                               the model artefacts parameters.
        Returns:
            list : Returns a list of dictionary with the predicted response.
        """
        start = timer()
        self.context = context
        self.initialize(context)
        data_preprocess = self.preprocess()
        inferred = self.inference(data_preprocess)
        output = self.postprocess(inferred)
        print("converting to seg.nrrd..")
        # self.convert_nifti_to_nrrd()
        print("Inference Done and saved prediction")
        end = timer()
        print(f"Time taken : {end - start}")
        return [{"Predicition": "Done", "Pred Path": self.output_dir}]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint_path",
        help="paths to model checkpoint file, or neptune run id",
    )

    parser.add_argument(
        "--input_file",
        type=Path,
        help="path to input CT file",
    )
    parser.add_argument(
        "--pt_file",
        type=Path,
        # action="append",
        help="path to input PT file",
        required=False,
    )

    parser.add_argument(
        "--prediction_save",
        type=Path,
        help="Directory to save predicted labels to, defaults to path in loaded model hparams",
    )
    parser.add_argument(
        "--predict_aug",
        action="store_true",
        help="Use test time augmentation flips",
    )
    parser.add_argument(
        "--softmax",
        action="store_true",
        help="Save softmax values instead of binary mask",
    )
    parser.add_argument(
        "--organ_name",
        type=str,
        help="organ name",
    )
    parser.add_argument("--fold", type=int, help="fold", default=0)
    config = parser.parse_args()
    return config


if __name__ == "__main__":
    config = parse_args()
    config_vars = vars(config)
    inference_model = BAMFnnUNetInference()
    inference_model.handle(config.checkpoint_path, config)
