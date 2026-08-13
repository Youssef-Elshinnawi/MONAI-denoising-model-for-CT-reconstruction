"""
Inference-time DICOM loading and preprocessing for the trained
ResidualDenoiser. Unlike dicom_loader.py's load_patient_pair() (which needs
paired full/low training data), this operates on a single real series with
no ground truth, and keeps the original per-slice pydicom Datasets so the
output DICOM can be written with correct headers.
"""

import os
import pydicom
import numpy as np
import torch
from monai.transforms import ScaleIntensityRange

# Must match dataset.py's build_transforms() exactly -- this is the same
# windowing the model was trained on.
HU_WINDOW_MIN = -1000
HU_WINDOW_MAX = 400


def load_series_for_inference(series_dir: str) -> tuple[list[pydicom.Dataset], np.ndarray]:
    """
    Read every .dcm in `series_dir`, sorted by InstanceNumber (same rule as
    dicom_loader.py -- filename order is not reliable).

    Returns: (sorted_datasets, hu_volume) where hu_volume has shape
    (num_slices, rows, cols), float32, in HU.
    """
    dcm_paths = [
        os.path.join(series_dir, f)
        for f in os.listdir(series_dir)
        if f.endswith(".dcm")
    ]

    datasets = [pydicom.dcmread(path) for path in dcm_paths]
    datasets.sort(key=lambda ds: ds.InstanceNumber)

    hu_slices = []
    for ds in datasets:
        # DSfloat (pydicom's float subclass) upcasts float32 -> float64 on
        # multiply per numpy's type promotion rules -- the trailing cast is
        # load-bearing, not redundant with the one on pixel_array.
        hu = ds.pixel_array.astype(np.float32) * ds.RescaleSlope + ds.RescaleIntercept
        hu_slices.append(hu.astype(np.float32))

    hu_volume = np.stack(hu_slices, axis=0)

    return datasets, hu_volume


def preprocess_for_model(hu_volume: np.ndarray) -> np.ndarray:
    """
    Apply the same HU windowing used in training (clip to
    [HU_WINDOW_MIN, HU_WINDOW_MAX], rescale to [0,1]), using MONAI's
    non-dict ScaleIntensityRange (same transform class as dataset.py's
    ScaleIntensityRanged, just without the dict-key wrapper).

    hu_volume: (num_slices, rows, cols) float32 HU array.
    Returns: same shape, float32, values in [0,1].
    """
    transform = ScaleIntensityRange(a_min=HU_WINDOW_MIN, a_max=HU_WINDOW_MAX, b_min=0.0, b_max=1.0, clip=True)
    normalized = transform(hu_volume)

    return np.asarray(normalized)


def run_inference(model, normalized_volume: np.ndarray, device, batch_size: int = 8) -> np.ndarray:
    """
    Run the model over `normalized_volume` (num_slices, H, W), values in
    [0,1], in chunks of `batch_size` slices at a time.

    Returns: denoised volume, same shape as input, values in [0,1].
    """
    model.eval()  # TODO: set the model to eval mode (matches evaluate.py)

    num_slices = normalized_volume.shape[0]
    output_volume = np.zeros_like(normalized_volume)

    # TODO: wrap the loop body below in `with torch.no_grad():` -- no
    # gradients needed for inference, saves memory (matches evaluate.py)
    with torch.no_grad():
        for start in range(0, num_slices, batch_size):
            end = min(start + batch_size, num_slices)

            chunk = normalized_volume[start:end]  # shape (chunk_size, H, W)

            # TODO: torch.from_numpy(chunk).float().unsqueeze(1).to(device)
            # -- unsqueeze(1) inserts the channel dim: (chunk_size, H, W) -> (chunk_size, 1, H, W)
            chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(1).to(device)

            output_tensor = model(chunk_tensor)  # TODO: pass chunk_tensor through the model

            # TODO: output_tensor.squeeze(1).cpu().numpy() -- remove the channel
            # dim back to (chunk_size, H, W) and bring the result back to numpy
            output_chunk = output_tensor.squeeze(1).cpu().numpy()

            output_volume[start:end] = output_chunk

    return output_volume


def postprocess_to_hu(denoised_normalized: np.ndarray) -> np.ndarray:
    """
    Inverse of preprocess_for_model(): clamp to [0,1] (the model's raw
    residual output can fall slightly outside this range -- see notes),
    then map [0,1] back to the [HU_WINDOW_MIN, HU_WINDOW_MAX] HU range.

    Returns: float32 HU array, same shape as input.
    """
    clamped = np.clip(denoised_normalized, 0.0, 1.0)  # TODO: np.clip(denoised_normalized, 0.0, 1.0)

    # TODO: the exact inverse of preprocess_for_model's linear rescale:
    #   hu = clamped * (HU_WINDOW_MAX - HU_WINDOW_MIN) + HU_WINDOW_MIN
    hu_volume = clamped * (HU_WINDOW_MAX - HU_WINDOW_MIN) + HU_WINDOW_MIN

    return hu_volume.astype(np.float32)


def write_denoised_dicom(datasets: list[pydicom.Dataset], hu_volume: np.ndarray, output_dir: str) -> None:
    """
    Write one new DICOM file per slice into `output_dir`, reusing each
    slice's original headers (patient info, geometry, etc.) but with the
    denoised pixel data and fresh series/instance UIDs (this is a genuinely
    different derived series, not an overwrite of the input).
    """
    os.makedirs(output_dir, exist_ok=True)

    new_series_uid = pydicom.uid.generate_uid()  # one shared UID for the whole output series

    for i, ds in enumerate(datasets):
        hu_slice = hu_volume[i]

        # TODO: invert the per-slice HU conversion from load_series_for_inference:
        #   pixel_value = (hu_slice - ds.RescaleIntercept) / ds.RescaleSlope
        # then round to the nearest integer and cast to ds.pixel_array.dtype
        # (the original storage dtype, e.g. int16) so PixelData stays valid.
        new_pixels = np.round((hu_slice - ds.RescaleIntercept) / ds.RescaleSlope).astype(ds.pixel_array.dtype)

        ds.PixelData = new_pixels.tobytes()
        ds.SeriesInstanceUID = new_series_uid
        ds.SOPInstanceUID = pydicom.uid.generate_uid()  # TODO: pydicom.uid.generate_uid() -- a fresh UID per slice
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.SeriesDescription = f"{getattr(ds, 'SeriesDescription', '')} - AI Denoised".strip(" -")

        output_path = os.path.join(output_dir, f"denoised_{i:04d}.dcm")
        ds.save_as(output_path)


if __name__ == "__main__":
    from model import ResidualDenoiser

    SERIES_DIR = "/Users/youssef/MONAI-denoising/frontier-app/test/sample_input"
    CHECKPOINT_PATH = "/Users/youssef/MONAI-denoising/checkpoints/best_model_run2_100ep.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on: {device}")

    datasets, hu_volume = load_series_for_inference(SERIES_DIR)
    normalized = preprocess_for_model(hu_volume)

    print(f"num slices: {len(datasets)}")
    print(f"hu_volume shape: {hu_volume.shape}, HU range: [{hu_volume.min():.1f}, {hu_volume.max():.1f}]")
    print(f"normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")

    model = ResidualDenoiser().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    denoised = run_inference(model, normalized, device)
    print(f"denoised shape: {denoised.shape}, range: [{denoised.min():.3f}, {denoised.max():.3f}]")

    denoised_hu = postprocess_to_hu(denoised)
    print(f"denoised HU range: [{denoised_hu.min():.1f}, {denoised_hu.max():.1f}]")

    OUTPUT_DIR = "/Users/youssef/MONAI-denoising/frontier-app/test/sample_output"
    write_denoised_dicom(datasets, denoised_hu, OUTPUT_DIR)
    print(f"wrote {len(datasets)} denoised DICOM files to {OUTPUT_DIR}")
