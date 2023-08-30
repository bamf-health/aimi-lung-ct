import pydicom
import subprocess
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd
import re
from tqdm.auto import tqdm
from pprint import pprint


class DcmError:
    def __init__(self, msg):
        tokens = msg.split(" - ")
        if len(tokens) == 3:
            self.t, self.path_to_attribute, self.message = tokens
            self.value = None
        elif len(tokens) == 4:
            self.t, self.path_to_attribute, self.message, self.value = tokens
        elif len(tokens) == 5:
            self.t, self.path_to_attribute, a, b, self.value = tokens
            self.message = f"{a} - {b}"
        else:
            raise ValueError(f"Error parsing error message: {msg}")

    def __repr__(self):
        return f"DcmError({self.path_to_attribute}, {self.message}, {self.value})"


class DcmBundle:
    def __init__(self, dicom_path: Path):
        self.dicom_path = dicom_path
        self.ds = pydicom.dcmread(dicom_path)

        # find errors
        cmd = ["dciodvfy", "-new", str(dicom_path)]
        proc = subprocess.run(cmd, capture_output=True)
        errs = proc.stderr.decode("utf-8").splitlines()
        self.errs = [DcmError(x) for x in errs if x.startswith("Error")]

    def fix(self):
        for err in self.errs:
            self._fix(err)

    def _fix(self, err: DcmError):
        if err.path_to_attribute.startswith("</ProcedureCodeSequence(0008,1032)"):
            # might have been removed already
            if hasattr(self.ds, "ProcedureCodeSequence"):
                del self.ds.ProcedureCodeSequence  # remove the sequence
        elif err.path_to_attribute == "</Laterality(0020,0060)>":
            self.ds.Laterality = ""
        elif err.path_to_attribute == "</PatientSex(0010,0040)[1]>":
            self.ds.PatientSex = ""
        elif err.path_to_attribute == "</PatientAge(0010,1010)[1]>":
            try:
                self.ds.PatientAge = f"{int(self.ds.PatientAge):03d}Y"
            except ValueError:
                del self.ds.PatientAge
        elif err.path_to_attribute == "</Manufacturer(0008,0070)>":
            self.ds.Manufacturer = ""
        elif err.path_to_attribute == "</ImageType(0008,0008)>":
            if self.ds.Modality == "CT":
                self.ds.ImageType = ""
            else:
                raise NotImplementedError(
                    f"fix for {err.path_to_attribute} not implemented for modality: {self.ds.Modality}"
                )
        elif (
            err.path_to_attribute
            == "</LongitudinalTemporalInformationModified(0028,0303)>"
        ):
            if hasattr(self.ds, "LongitudinalTemporalInformationModified"):
                del self.ds.LongitudinalTemporalInformationModified
        elif err.path_to_attribute.startswith("</RequestAttributesSequence(0040,0275)"):
            if hasattr(self.ds, "RequestAttributesSequence"):
                del self.ds.RequestAttributesSequence
        elif err.path_to_attribute == "</ClinicalTrialSponsorName(0012,0010)>":
            self.ds.ClinicalTrialSponsorName = "UNKNOWN"
        elif err.path_to_attribute == "</ClinicalTrialSubjectID(0012,0040)>":
            self.ds.ClinicalTrialSubjectID = "UNKNOWN"
        elif err.path_to_attribute == "</ClinicalTrialSubjectReadingID(0012,0042)>":
            self.ds.ClinicalTrialSubjectReadingID = "UNKNOWN"
        elif err.path_to_attribute.startswith(
            "</PatientOrientationCodeSequence(0054,0410)[1]"
        ):
            self.ds.PatientOrientationCodeSequence = []
        elif (
            err.path_to_attribute == "</PatientPosition(0018,5100)>"
            and err.message
            == "Shall not be present when PatientOrientationCodeSequence is present"
            and hasattr(self.ds, "PatientOrientationCodeSequence")
            and len(self.ds.PatientOrientationCodeSequence) > 0
        ):
            del self.ds.PatientPosition
        elif (
            err.path_to_attribute == "</NumberOfTimeSlices(0054,0101)>"
            and err.message
            == "Attribute present when condition unsatisfied (which may not be present otherwise) for Type 1C Conditional"
        ):
            del self.ds.NumberOfTimeSlices
        elif (
            err.path_to_attribute == "</FrameTime(0018,1063)>"
            and err.message
            == "Attribute present when condition unsatisfied (which may not be present otherwise) for Type 1C Conditional"
        ):
            del self.ds.FrameTime
        elif (
            err.path_to_attribute == "</TriggerTime(0018,1060)>"
            and err.message
            == "Attribute present when condition unsatisfied (which may not be present otherwise) for Type 1C Conditional"
        ):
            del self.ds.TriggerTime
        elif err.path_to_attribute.startswith(
            "</AcquisitionContextSequence(0040,0555)"
        ):
            if hasattr(self.ds, "AcquisitionContextSequence"):
                del self.ds.AcquisitionContextSequence
        elif err.path_to_attribute.startswith(
            "</RadiopharmaceuticalInformationSequence(0054,0016)[1]/RadiopharmaceuticalCodeSequence(0054,0304)"
        ):
            if hasattr(self.ds, "RadiopharmaceuticalInformationSequence") and hasattr(
                self.ds.RadiopharmaceuticalInformationSequence[0],
                "RadiopharmaceuticalCodeSequence",
            ):
                del self.ds.RadiopharmaceuticalInformationSequence[
                    0
                ].RadiopharmaceuticalCodeSequence
        elif err.path_to_attribute == "</DecayFactor(0054,1321)>":
            # fake data is not ideal, but since we are using this for segmentation dicom metadata, it should be fine
            self.ds.DecayFactor = 1.0
        elif err.path_to_attribute == "</RescaleSlope(0028,1053)[1]>":
            # fake data is not ideal, but since we are using this for segmentation dicom metadata, it should be fine
            self.ds.RescaleSlope = 1.0
        elif err.path_to_attribute == "</PhotometricInterpretation(0028,0004)[1]>":
            # value shouldn't matter since this is just a reference for the segmentation dicom
            self.ds.PhotometricInterpretation = "MONOCHROME2"

        # troublesome private tags
        elif err.path_to_attribute == '</(0013,1010,"CTP")>':
            del self.ds[0x0013, 0x1010]
        elif m := re.match(
            r'<\/\(([\da-f]{4}),([\da-f]{4}),"GEMS_PETD_01"\)>', err.path_to_attribute
        ):
            a = int(m.group(1), 16)
            b = int(m.group(2), 16)
            del self.ds[a, b]

        else:
            raise NotImplementedError(
                f"fix for '{err.t} - {err.path_to_attribute} - {err.message} - {err.value}' not implemented for modality: {self.ds.Modality}"
            )

    def test(self):
        with TemporaryDirectory() as tempdir:
            tempdir = Path(tempdir)
            tempdir.mkdir(exist_ok=True)
            temp_dcm = tempdir / "temp.dcm"
            self.ds.save_as(temp_dcm)
            cmd = ["dciodvfy", "-new", str(temp_dcm)]
            proc = subprocess.run(cmd, capture_output=True)
            errs = proc.stderr.decode("utf-8").split("\n")
            errs = [DcmError(x) for x in errs if x.startswith("Error")]
            return errs


def fix_dicom_dir(dicom_dir: Path, output_dir: Path):
    """Scan the dicom files in dicom_dir and check with dciodvfy for errors. If any errors are found, attempt to fix them and write the fixed dicom files to output_dir.

    Args:
        dicom_dir (Path): input directory of dicom files
        output_dir (Path): writes fixed dicom files to this directory if any fixes are required

    Returns:
        _type_: output_dir if any fixes were required, otherwise dicom_dir
    """
    dcm_files = [x for x in dicom_dir.rglob("*") if pydicom.misc.is_dicom(x)]

    dss = [DcmBundle(dcm_file) for dcm_file in dcm_files]

    error_cnt = sum([len(x.errs) for x in dss])

    if error_cnt == 0:
        return dicom_dir

    for x in dss:
        out_file = output_dir / x.dicom_path.relative_to(dicom_dir)
        x.fix()
        if len(x.errs):
            out_file.parent.mkdir(exist_ok=True, parents=True)
            x.ds.save_as(out_file)

    return output_dir


def fix_it_all():
    logged_errors = set()
    df = pd.read_csv("/home/vanossj/projects/aimi-idc-data/tasks/non-complient-dcm.csv")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if i < 249:
            continue
        dcm_dir = Path(row["dcm_dir"])
        dcm_files = [x for x in dcm_dir.rglob("*") if pydicom.misc.is_dicom(x)]
        for dcm_file in dcm_files:
            x = DcmBundle(dcm_file)
            try:
                x.fix()
            except NotImplementedError as e:
                msg = str(e)
                if msg not in logged_errors:
                    logged_errors.add(str(e))
                    print(e)
                continue
            except AttributeError as e:
                print(dcm_file)
                pprint(x.errs)
                raise e


if __name__ == "__main__":
    fix_it_all()