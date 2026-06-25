from pathlib import Path
import os

PROJECT_NAME = "AIUB_Turtlers_FictiPay_Survival"
RANDOM_SEED = 42
N_SPLITS = int(os.environ.get("N_SPLITS", "5"))
TRAIN_CHUNK_SIZE = int(os.environ.get("TRAIN_CHUNK_SIZE", "20"))
USE_GPU = os.environ.get("USE_GPU", "1") == "1"
DATA_DIR = Path(os.environ.get("FICTIPAY_DATA_DIR", r"C:\Users\rhyth\OneDrive\Desktop\Datathon\public"))
WORK_DIR = Path(os.environ.get("FICTIPAY_WORK_DIR", str(DATA_DIR.parent / "work_survival_scripts")))
FEATURES_PATH = WORK_DIR / "features.parquet"
PREDICTIONS_PATH = WORK_DIR / "model_predictions.npz"
SUBMISSION_PATH = WORK_DIR / "AIUB_Turtlers_survival_submission.csv"
OBSERVATION_END = "2024-03-31"
OBSERVATION_END_EXCLUSIVE = "2024-04-01"
HORIZONS = [30, 60, 90]
WINDOWS = [1, 3, 7, 14, 30, 60, 90]
TOP_TRX_TYPES = 12
