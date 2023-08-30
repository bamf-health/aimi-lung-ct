#!/usr/bin/env python3
import argparse
import shutil
import sys
from pathlib import Path
import pydicom
import os


class DotDict(dict):
    def __getattr__(self, attr):
        return self.get(attr)


def get_path(*paths):
    """
    Concatenate paths to create a complete file path.

    Args:
        *paths: Variable number of path segments.

    Returns:
        str: Complete file path.
    """
    return os.path.join(*paths)


def copy_to_series_dir(input_file, dcm_dir, output_dir):
    series_uid = None
    try:
        for f in dcm_dir.glob("*"):
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            series_uid = ds.SeriesInstanceUID
            break
    except:
        print("Failed to read dicom files")
        sys.exit(1)

    assert series_uid is not None

    output_path = output_dir / series_uid / input_file.name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(input_file, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    parser.add_argument("dcm_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    copy_to_series_dir(
        input_file=args.input_file,
        dcm_dir=args.dcm_dir,
        output_dir=args.output_dir
    )
