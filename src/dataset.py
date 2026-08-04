"""
Patient-level train/val/test split and a MONAI Dataset that yields paired
(low-dose, full-dose) 2D slices with HU windowing applied.
"""

import os
from typing import Sequence
from monai.transforms import ScaleIntensityRanged, Compose, EnsureChannelFirstd
from monai.data import Dataset

from dicom_loader import load_patient_pair

ALL_PATIENTS = ["C002", "C004", "C012", "C016", "C021", "C027", "C030", "C050", "C052", "C067"]

# Split by patient, not by slice -- adjacent slices within a patient are highly
# correlated, so a slice-level split would leak anatomy between train/val/test.
TRAIN_PATIENTS: list[str] = ["C002", "C004", "C012", "C016", "C021", "C027"]
VAL_PATIENTS: list[str] = ["C030", "C050"]
TEST_PATIENTS: list[str] = ["C052", "C067"]


def build_slice_dicts(patient_ids: Sequence[str], data_root: str) -> list[dict]:
    """
    Load every patient's (full, low) volumes and flatten them into one dict
    per 2D slice: {"low": <512x512 float32 array>, "full": <512x512 float32 array>}.
    """
    slice_dicts = []

    for patient_id in patient_ids:
        patient_dir = os.path.join(data_root, patient_id)
        full_vol, low_vol = load_patient_pair(patient_dir)

        num_slices = full_vol.shape[0]
        for i in range(num_slices):
            slice_dicts.append({"low": low_vol[i], "full": full_vol[i]})

    return slice_dicts


def build_transforms() -> Compose:
    """
    EnsureChannelFirstd: adds a channel dim (512,512) -> (1,512,512) for 2D conv layers.
    ScaleIntensityRanged: clips to the [-1000, 400] HU chest window and rescales to [0, 1].
    """
    return Compose([
        EnsureChannelFirstd(keys=["low", "full"], channel_dim="no_channel"),
        ScaleIntensityRanged(keys=["low", "full"], a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True),
    ])


def build_dataset(patient_ids: Sequence[str], data_root: str) -> Dataset:
    slice_dicts = build_slice_dicts(patient_ids, data_root)
    transforms = build_transforms()
    return Dataset(data=slice_dicts, transform=transforms)


if __name__ == "__main__":
    DATA_ROOT = "/content/drive/MyDrive/MONAI-denoising/data/raw"

    train_ds = build_dataset(TRAIN_PATIENTS, DATA_ROOT)
    print(f"train dataset size (num slices): {len(train_ds)}")

    sample = train_ds[0]
    print(f"sample['low'] shape: {sample['low'].shape}, dtype: {sample['low'].dtype}")
    print(f"sample['low'] range: [{sample['low'].min():.3f}, {sample['low'].max():.3f}]")
    print(f"sample['full'] range: [{sample['full'].min():.3f}, {sample['full'].max():.3f}]")
