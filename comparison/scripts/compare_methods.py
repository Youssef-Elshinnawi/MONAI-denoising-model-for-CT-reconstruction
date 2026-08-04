"""
4-way comparison -- FBP baseline, SART+TV, FBP+MONAI-denoise, and
SART+TV+MONAI-denoise -- against the same real CT slice ground truth, under
a shared simulated low-dose noise model (see comparison/cpp/src/main_ct_compare.cpp).

Runs entirely locally on CPU: the .bin files come from the C++ pipeline in
comparison/cpp (run `make build/ct_compare` there first), and the MONAI
checkpoint was trained on Colab and downloaded to checkpoints/.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from monai.networks.nets import UNet
from monai.metrics import PSNRMetric, SSIMMetric


class ResidualDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2,
        )

    def forward(self, x):
        return x + self.unet(x)


def compute_nrmse(reconstruction: np.ndarray, phantom: np.ndarray) -> float:
    """
    Same formula as Project 2's validation.cpp: rescale reconstruction to the
    phantom's min/max range first (back-projection doesn't guarantee matching
    scale), then RMSE normalized by the phantom's dynamic range.
    """
    recon_min, recon_max = reconstruction.min(), reconstruction.max()
    phantom_min, phantom_max = phantom.min(), phantom.max()

    recon_scaled = (reconstruction - recon_min) / (recon_max - recon_min) * (phantom_max - phantom_min) + phantom_min

    rmse = np.sqrt(np.mean((recon_scaled - phantom) ** 2))
    return rmse / (phantom_max - phantom_min)


N = 256
FIGURES_DIR = "/Users/youssef/MONAI-denoising/comparison/cpp/figures/ct_compare"
CHECKPOINT_PATH = "/Users/youssef/MONAI-denoising/checkpoints/best_model_run2_100ep.pt"

phantom = np.fromfile(os.path.join(FIGURES_DIR, "phantom.bin"), dtype=np.float64).reshape(N, N)
fbp_recon = np.fromfile(os.path.join(FIGURES_DIR, "fbp_recon.bin"), dtype=np.float64).reshape(N, N)
sart_tv_recon = np.fromfile(os.path.join(FIGURES_DIR, "sart_tv_recon_best.bin"), dtype=np.float64).reshape(N, N)

print("value ranges (ground truth / fbp / sart+tv):")
print(f"  phantom:   [{phantom.min():.3f}, {phantom.max():.3f}]")
print(f"  fbp_recon: [{fbp_recon.min():.3f}, {fbp_recon.max():.3f}]")
print(f"  sart_tv:   [{sart_tv_recon.min():.3f}, {sart_tv_recon.max():.3f}]")

model = ResidualDenoiser()
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
model.eval()

with torch.no_grad():
    fbp_tensor = torch.from_numpy(fbp_recon).float().clamp(0, 1).unsqueeze(0).unsqueeze(0)
    monai_output_tensor = model(fbp_tensor)
    monai_output = monai_output_tensor.squeeze().numpy()

    sart_tv_tensor = torch.from_numpy(sart_tv_recon).float().clamp(0, 1).unsqueeze(0).unsqueeze(0)
    sart_tv_monai_tensor = model(sart_tv_tensor)
    sart_tv_monai_output = sart_tv_monai_tensor.squeeze().numpy()

psnr_metric = PSNRMetric(max_val=1.0)
ssim_metric = SSIMMetric(spatial_dims=2, data_range=1.0)


def to_tensor(arr):
    return torch.from_numpy(arr).float().clamp(0, 1).unsqueeze(0).unsqueeze(0)


gt_tensor = to_tensor(phantom)

methods = {
    "FBP baseline\n(noisy)": fbp_recon,
    "SART+TV": sart_tv_recon,
    "FBP +\nMONAI denoise": monai_output,
    "SART+TV +\nMONAI denoise": sart_tv_monai_output,
}

results = {}
print()
print(f"{'method':<24} {'PSNR (dB)':>10} {'SSIM':>8} {'NRMSE':>8}")
for name, recon in methods.items():
    recon_tensor = to_tensor(recon)
    psnr = psnr_metric(recon_tensor, gt_tensor).item()
    ssim = ssim_metric(recon_tensor, gt_tensor).item()
    nrmse = compute_nrmse(recon, phantom)
    results[name] = {"psnr": psnr, "ssim": ssim, "nrmse": nrmse}
    print(f"{name.replace(chr(10), ' '):<24} {psnr:>10.3f} {ssim:>8.4f} {nrmse:>8.4f}")

# ---- categorical palette (fixed order, from the project's default palette) ----
COLOR_BLUE = "#2a78d6"
COLOR_ORANGE = "#eb6834"
COLOR_AQUA = "#1baf7a"
COLOR_YELLOW = "#eda100"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

method_colors = [COLOR_BLUE, COLOR_ORANGE, COLOR_AQUA, COLOR_YELLOW]

# ---- Figure 1: grayscale reconstructions (top row) + colored abs-diff maps
# with NRMSE underneath (bottom row) ----
from matplotlib.colors import LinearSegmentedColormap

# sequential blue ramp (light -> dark), per the dataviz skill's default
# sequential hue -- used for magnitude (the abs-diff maps), kept distinct in
# role from the categorical blue used for "FBP baseline" in the bar chart
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

panel_images = [phantom, fbp_recon, sart_tv_recon, monai_output, sart_tv_monai_output]
panel_titles = ["Ground truth\n(real slice)"] + list(methods.keys())

diff_maps = {name: np.abs(np.clip(recon, 0, 1) - phantom) for name, recon in methods.items()}
# Fixed shared scale (not the true max, which is stretched by a handful of
# extreme outlier pixels and washes out the real per-method contrast) -- 0.5
# comfortably covers the meaningful dynamic range (FBP's 90th-percentile
# error is ~0.67, SART+TV's is ~0.20) while still showing FBP as visibly
# worse than the others, which is the honest, correct relationship.
diff_vmax = 0.5

fig, axes = plt.subplots(2, 5, figsize=(20, 9), facecolor=SURFACE, constrained_layout=True)

for ax, img, title in zip(axes[0], panel_images, panel_titles):
    ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=11, color=INK_PRIMARY)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

# bottom-left (under ground truth): no diff against itself -- leave blank
axes[1, 0].axis("off")
axes[1, 0].text(
    0.5, 0.5, "(ground truth --\nno diff to itself)",
    ha="center", va="center", fontsize=10, color=INK_MUTED,
    transform=axes[1, 0].transAxes,
)

im = None
for ax, name in zip(axes[1, 1:], methods.keys()):
    im = ax.imshow(diff_maps[name], cmap=SEQUENTIAL_BLUE, vmin=0, vmax=diff_vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    nrmse = results[name]["nrmse"]
    ax.set_xlabel(f"NRMSE = {nrmse:.4f}", fontsize=10, color=INK_SECONDARY)

# one shared colorbar for the diff row (magnitude scale, shared across panels)
cbar = fig.colorbar(im, ax=axes[1, 1:].tolist(), shrink=0.8, pad=0.02, extend="max", label="|reconstruction - ground truth|")
cbar.ax.yaxis.label.set_color(INK_SECONDARY)
cbar.ax.tick_params(colors=INK_MUTED)

fig.patch.set_facecolor(SURFACE)
image_panel_path = "/Users/youssef/MONAI-denoising/outputs/comparison_all_methods.png"
plt.savefig(image_panel_path, dpi=150, facecolor=SURFACE)
print(f"\nsaved image comparison (with colored diff maps + NRMSE) to {image_panel_path}")
plt.close(fig)

# ---- Figure 2: PSNR and SSIM bar charts, side by side (no dual-axis) ----
labels = list(methods.keys())
psnr_values = [results[name]["psnr"] for name in labels]
ssim_values = [results[name]["ssim"] for name in labels]

fig, (ax_psnr, ax_ssim) = plt.subplots(1, 2, figsize=(13, 5), facecolor=SURFACE)

for ax, values, title, fmt in [
    (ax_psnr, psnr_values, "PSNR (dB) -- higher is better", "{:.2f}"),
    (ax_ssim, ssim_values, "SSIM -- higher is better", "{:.3f}"),
]:
    ax.set_facecolor(SURFACE)
    bars = ax.bar(range(len(labels)), values, color=method_colors, width=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9, color=INK_SECONDARY)
    ax.set_title(title, fontsize=12, color=INK_PRIMARY)
    ax.tick_params(axis="y", colors=INK_MUTED, labelsize=9)
    ax.grid(axis="y", color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine_name, spine in ax.spines.items():
        spine.set_visible(spine_name == "bottom")
        if spine_name == "bottom":
            spine.set_color(INK_MUTED)

    # direct value labels above each bar
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            fmt.format(value),
            ha="center", va="bottom", fontsize=9, color=INK_PRIMARY,
        )

fig.patch.set_facecolor(SURFACE)
plt.tight_layout()
metrics_chart_path = "/Users/youssef/MONAI-denoising/outputs/comparison_metrics_chart.png"
plt.savefig(metrics_chart_path, dpi=150, facecolor=SURFACE)
print(f"saved PSNR/SSIM bar chart to {metrics_chart_path}")
