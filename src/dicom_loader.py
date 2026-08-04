"""
Loads a patient's full-dose / low-dose DICOM series into matched,
HU-corrected numpy volumes.

Expected data layout:
    <DATA_ROOT>/<patient_id>/full/*.dcm
    <DATA_ROOT>/<patient_id>/low/*.dcm
"""

import os
import pydicom
import numpy as np


def load_dicom_series_as_hu(series_dir: str) -> np.ndarray:
    """
    Load every .dcm file in `series_dir` into a single 3D HU volume,
    slices ordered by DICOM InstanceNumber (ascending).

    Returns: float32 array of shape (num_slices, rows, cols).
    """
    dcm_paths = [
        os.path.join(series_dir, f)
        for f in os.listdir(series_dir)
        if f.endswith(".dcm")
    ]

    datasets = [pydicom.dcmread(path) for path in dcm_paths]

    # InstanceNumber order doesn't necessarily match filename order.
    datasets.sort(key=lambda ds: ds.InstanceNumber)

    slices = []
    for ds in datasets:
        hu = ds.pixel_array.astype(np.float32) * ds.RescaleSlope + ds.RescaleIntercept
        slices.append(hu)

    return np.stack(slices, axis=0)


def load_patient_pair(patient_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a patient's full-dose and low-dose volumes and verify they're
    slice-registered (same slice count/shape).

    Returns: (full_dose_volume, low_dose_volume), both float32 HU arrays.
    """
    full_dir = os.path.join(patient_dir, "full")
    low_dir = os.path.join(patient_dir, "low")

    full_vol = load_dicom_series_as_hu(full_dir)
    low_vol = load_dicom_series_as_hu(low_dir)

    if full_vol.shape != low_vol.shape:
        raise ValueError(
            f"shape mismatch for {patient_dir}: full={full_vol.shape} vs low={low_vol.shape}"
        )

    return full_vol, low_vol


if __name__ == "__main__":
    DATA_ROOT = "/content/drive/MyDrive/MONAI-denoising/data/raw/C002"
    full_vol, low_vol = load_patient_pair(DATA_ROOT)

    print(f"full-dose volume shape: {full_vol.shape}, dtype: {full_vol.dtype}")
    print(f"low-dose volume shape:  {low_vol.shape}, dtype: {low_vol.dtype}")
    print(f"full-dose HU range: [{full_vol.min():.1f}, {full_vol.max():.1f}]")
    print(f"low-dose HU range:  [{low_vol.min():.1f}, {low_vol.max():.1f}]")
