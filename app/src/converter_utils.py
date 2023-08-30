#!/usr/bin/env python3
import argparse
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
import pydicom
import SimpleITK as sitk
import os
from fix_dicom import fix_dicom_dir


class DicomToNiiConverter:
    def __init__(self) -> None:
        pass

    def dcm_to_niix(self, dcm_dir: Path, nii_path: Path):
        """uses dcm2niix to convert a series of dicom files to a nifti file"""
        dcm_dir = Path(dcm_dir)
        nii_path = Path(nii_path)
        with TemporaryDirectory() as tmpdir:
            args = [
                "dcm2niix",
                "-o",
                tmpdir,
                "-z",
                "y",
                str(dcm_dir.resolve()),
            ]
            subprocess.run(args, check=True)

            nii_files = list(Path(tmpdir).glob("*Eq_*.nii.gz"))
            if len(nii_files) > 1:
                raise ValueError(f"Expected 1 Eq_*.nii.gz file, found {len(nii_files)}")
            elif len(nii_files) == 1:
                shutil.move(nii_files[0], nii_path)
                return
            # no Eq images
            nii_files = list(Path(tmpdir).glob("*.nii.gz"))
            if len(nii_files) > 1:
                raise ValueError(f"Expected 1 *.nii.gz file, found {len(nii_files)}")
            elif len(nii_files) == 1:
                shutil.move(nii_files[0], nii_path)
                return
            raise ValueError(f"Expected 1 *.nii.gz file, found 0")


    def dcm_to_nii(self, dcm_dir: Path, nii_path: Path) -> bool:
        """uses SimpleITK to convert a series of dicom files to a nifti file"""
        # sort the files to hopefully make conversion a bit more reliable
        files = []
        dcm_dir = Path(dcm_dir)
        nii_path = Path(nii_path)
        for f in dcm_dir.glob("*.dcm"):
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            slicer_loc = ds.SliceLocation if hasattr(ds, "SlicerLocation") else 0
            files.append((slicer_loc, f))
        slices = sorted(files, key=lambda s: s[0])
        ordered_files = [x[1] for x in slices]

        with TemporaryDirectory() as tmp_dir:
            ptmp_dir = Path(tmp_dir)
            for i, f in enumerate(ordered_files):
                shutil.copy(f, ptmp_dir / f"{i}.dcm")
            try:
                # load in with SimpleITK
                reader = sitk.ImageSeriesReader()
                dicom_names = reader.GetGDCMSeriesFileNames(tmp_dir)
                reader.SetFileNames(dicom_names)
                image = reader.Execute()

                nii_path.parent.mkdir(parents=True, exist_ok=True)
                # save as nifti
                sitk.WriteImage(
                    image, str(nii_path.resolve()), useCompression=True, compressionLevel=9
                )
            except:
                return False
            return True


class NiiToDicomConverter:
    def __init__(self, dcmqi_package_path:str) -> None:
        dcmqi_bin_path = os.path.join(dcmqi_package_path, "bin/itkimage2segimage")
        self.itkimage2segimage_bin = (dcmqi_bin_path)

    def _convert_nii_to_dcm(
            self,
            nii_path: Path,
            dcm_ref_dir: Path,
            dcm_out_file: Path,
            dicom_seg_meta_json: Path,
            add_background_label: bool = False,
            ):
        
        assert dcm_ref_dir.exists(), dcm_ref_dir

        dcm_out_file = Path(dcm_out_file)
        dcm_out_file.parent.mkdir(parents=True, exist_ok=True)

        if add_background_label:
            # add background label, offset by 1
            with TemporaryDirectory() as temp_dir:
                temp_seg_file = Path(temp_dir) / "temp_seg.nii.gz"
                img = sitk.ReadImage(str(nii_path))
                img += 1
                sitk.WriteImage(img, str(temp_seg_file))

                args = [
                    self.itkimage2segimage_bin,
                    "--skip",
                    "--inputImageList",
                    str(temp_seg_file),
                    "--inputDICOMDirectory",
                    str(dcm_ref_dir),
                    "--outputDICOM",
                    str(dcm_out_file),
                    "--inputMetadata",
                    str(dicom_seg_meta_json),
                ]

                subprocess.run(args, check=True)
        else:
            args = [
                self.itkimage2segimage_bin,
                "--skip",
                "--inputImageList",
                str(nii_path),
                "--inputDICOMDirectory",
                str(dcm_ref_dir),
                "--outputDICOM",
                str(dcm_out_file),
                "--inputMetadata",
                str(dicom_seg_meta_json),
            ]

            print(" ".join(args))
            subprocess.run(args, check=True)

    def convert_nii_to_dcm(
            self,
            nii_path: Path,
            dcm_ref_dir: Path,
            dcm_out_file: Path,
            dicom_seg_meta_json: Path,
            add_background_label: bool = False
            ):
        
        status = True
        try:
            self._convert_to_dicom_seg(
                nii_path,
                dcm_ref_dir,
                dcm_out_file,
                dicom_seg_meta_json,
                add_background_label=add_background_label
            )
        except Exception as e:
            status = False
        return status
    
        if status:
            return

        # # fix the dicom files, and try again
        # with TemporaryDirectory() as fixed_dcm_dir:
        #     real_dcm_dir = fix_dicom_dir(dcm_ref_dir, Path(fixed_dcm_dir))
        #     self._convert_to_dicom_seg(
        #         nii_path,
        #         real_dcm_dir,
        #         dcm_out_file,
        #         dicom_seg_meta_json,
        #         add_background_label=add_background_label
        #     )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert dicom files to nifti file. Supports single series"
    )
    parser.add_argument(
        "dcm_dir", type=Path, help="input directory containing dicom files"
    )
    parser.add_argument("nii_path", type=Path, help="output directory for nifti files")
    parser.add_argument(
        "--niix",
        action="store_true",
        help="use dcm2niix instead of SimpleITK for conversion",
    )
    args = parser.parse_args()

    converter = DicomToNiiConverter()
    if args.niix:
        converter.dcm_to_niix(args.dcm_dir, args.nii_path)
    else:
        converter.dcm_to_nii*(args.dcm_dir, args.nii_path)
