"""
Evaluates the trained ResidualDenoiser on held-out test patients: PSNR/SSIM
of (low-dose vs full-dose) baseline against (model output vs full-dose).

Uses MONAI's own metrics (torch-native) instead of scikit-image -- avoids a
numpy/scikit-image version conflict encountered on Colab.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from monai.metrics import PSNRMetric, SSIMMetric

from dataset import build_dataset, TEST_PATIENTS
from model import ResidualDenoiser


def evaluate(model, test_ds, device) -> dict:
    """
    Returns a dict of per-slice score lists: {"baseline_psnr": [...],
    "model_psnr": [...], "baseline_ssim": [...], "model_ssim": [...]}.
    """
    model.eval()
    loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    psnr_metric = PSNRMetric(max_val=1.0)
    ssim_metric = SSIMMetric(spatial_dims=2, data_range=1.0)

    results = {"baseline_psnr": [], "model_psnr": [], "baseline_ssim": [], "model_ssim": []}

    with torch.no_grad():
        for batch in loader:
            low = batch["low"].to(device)
            full = batch["full"].to(device)

            output = model(low)

            baseline_psnr = psnr_metric(low, full)
            model_psnr = psnr_metric(output, full)
            baseline_ssim = ssim_metric(low, full)
            model_ssim = ssim_metric(output, full)

            results["baseline_psnr"].extend(baseline_psnr.flatten().cpu().tolist())
            results["model_psnr"].extend(model_psnr.flatten().cpu().tolist())
            results["baseline_ssim"].extend(baseline_ssim.flatten().cpu().tolist())
            results["model_ssim"].extend(model_ssim.flatten().cpu().tolist())

    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_ROOT = "/content/drive/MyDrive/MONAI-denoising/data/raw"
    CHECKPOINT_PATH = "/content/drive/MyDrive/MONAI-denoising/checkpoints/best_model.pt"

    test_ds = build_dataset(TEST_PATIENTS, DATA_ROOT)

    model = ResidualDenoiser().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    results = evaluate(model, test_ds, device)

    for key, values in results.items():
        arr = np.array(values)
        print(f"{key}: mean={arr.mean():.4f}  std={arr.std():.4f}")

    print()
    print(f"PSNR improvement: {np.mean(results['model_psnr']) - np.mean(results['baseline_psnr']):.4f} dB")
    print(f"SSIM improvement: {np.mean(results['model_ssim']) - np.mean(results['baseline_ssim']):.4f}")
