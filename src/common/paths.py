"""
Project-wide paths and hardware settings.

Every script used to work out the project folder on its own with
os.path.join(here, "..", ".."). Keeping it here means a folder move or a new
drone address is a one-line change instead of an edit in ten files.
"""
import os

# src/common/paths.py -> two folders up is the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

SCENE_XML = os.path.join(PROJECT_ROOT, "Assets", "bitcraze_crazyflie_2", "scene.xml")  # MuJoCo world with the drone
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")  # trained policies, one folder per run
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")      # TensorBoard curves, eval results, CSV logs

# cflib stores the drone's parameter and log tables here so later connections are faster.
# It is regenerated automatically and is ignored by git.
CFLIB_CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

# Radio address of the Crazyflie. The April 2026 flight scripts used ...E703;
# the older calibration scripts used ...E7E7. Change it here if your drone differs.
DEFAULT_URI = "radio://0/80/2M/E7E7E7E703"


def model_dir(run_name):
    """Folder that holds best_model.zip and vecnormalize_best.pkl for a training run."""
    return os.path.join(MODELS_DIR, run_name)


def log_dir(run_name):
    """Folder that holds the TensorBoard logs and evaluations.npz for a training run."""
    return os.path.join(LOGS_DIR, run_name)
