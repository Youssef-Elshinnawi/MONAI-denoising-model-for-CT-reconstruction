"""
Entrypoint for the Frontier processing unit container. Reads a DICOM series
from /mnt/input, denoises it via the trained ResidualDenoiser, and writes
the result to /mnt/output.

Mount-path convention per Frontier's documented Linux container onboarding:
docs.frontier.api.teamplay.siemens-healthineers.com/docs/Overview/Intro
"""

import os
import sys
import json
import logging
import torch

from inference import (
    load_series_for_inference,
    preprocess_for_model,
    run_inference,
    postprocess_to_hu,
    write_denoised_dicom,
)
from model import ResidualDenoiser

INPUT_DIR = "/mnt/input"
OUTPUT_DIR = "/mnt/output"
CONFIG_DIR = "/mnt/config"
LOG_DIR = "/mnt/log"

DEFAULT_CHECKPOINT = "/app/checkpoints/best_model_run2_100ep.pt"
DEFAULT_BATCH_SIZE = 8


def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "processing.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_config() -> dict:
    """
    Reads the JSON config file from /mnt/config (the file the UI
    Configuration step -- Stage G -- produces). Falls back to defaults if
    no config file is present, so the container is still runnable standalone.
    """
    config_path = os.path.join(CONFIG_DIR, "config.json")
    if not os.path.exists(config_path):
        logging.warning(f"no config file found at {config_path}, using defaults")
        return {}

    with open(config_path) as f:
        return json.load(f)


def main():
    setup_logging()
    logging.info("starting CT denoising processing unit")

    try:
        config = load_config()
        checkpoint_path = config.get("checkpoint_path", DEFAULT_CHECKPOINT)
        batch_size = config.get("batch_size", DEFAULT_BATCH_SIZE)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info(f"loading checkpoint: {checkpoint_path}")
        model = ResidualDenoiser().to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        logging.info(f"reading input series from {INPUT_DIR}")
        datasets, hu_volume = load_series_for_inference(INPUT_DIR)

        normalized = preprocess_for_model(hu_volume)

        logging.info(f"running inference on {len(datasets)} slices")
        denoised = run_inference(model, normalized, device, batch_size)

        denoised_hu = postprocess_to_hu(denoised)

        logging.info(f"writing output to {OUTPUT_DIR}")
        write_denoised_dicom(datasets, denoised_hu, OUTPUT_DIR)

        logging.info("processing complete")

    except Exception:
        logging.exception("processing failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
