"""
Local test harness for entrypoint.py, run WITHOUT Docker -- sanity-checks
the Stage F wiring against real local test data before building the actual
container in Stage H.

Overrides the fixed /mnt/... paths with local test directories. This
override only exists here, for pre-Docker testing -- the real container
never does this; Docker's `-v host:/mnt/...` mounts are what make
/mnt/input etc. resolve to real data on the actual image.
"""

import os
import sys

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTIER_APP_DIR = os.path.dirname(TEST_DIR)
REPO_ROOT = os.path.dirname(FRONTIER_APP_DIR)

sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, FRONTIER_APP_DIR)

import entrypoint

entrypoint.INPUT_DIR = os.path.join(TEST_DIR, "sample_input")
entrypoint.OUTPUT_DIR = os.path.join(TEST_DIR, "sample_output")
entrypoint.CONFIG_DIR = os.path.join(TEST_DIR, "sample_config")
entrypoint.LOG_DIR = os.path.join(TEST_DIR, "sample_log")
entrypoint.DEFAULT_CHECKPOINT = os.path.join(REPO_ROOT, "checkpoints", "best_model_run2_100ep.pt")

entrypoint.main()
