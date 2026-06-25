from pathlib import Path
import os
import gc
import time
import numpy as np
import pandas as pd

try:
    import psutil
except Exception:
    psutil = None


def resolve_data_dir(data_dir):
    data_dir = Path(data_dir)
    if data_dir.exists() and (data_dir / "train_labels.csv").exists():
        return data_dir
    kaggle_root = Path("/kaggle/input")
    if kaggle_root.exists():
        matches = list(kaggle_root.rglob("train_labels.csv"))
        for m in matches:
            p = m.parent
            if (p / "test.csv").exists() and (p / "kyc.parquet").exists():
                return p
    raise FileNotFoundError(f"Dataset not found: {data_dir}")


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def memory_mb():
    if psutil is None:
        return 0.0
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def reduce_memory(df):
    for col in df.columns:
        if col == "ACCOUNT_ID":
            continue
        dtype = df[col].dtype
        if pd.api.types.is_float_dtype(dtype):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif pd.api.types.is_integer_dtype(dtype):
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def clean_numeric(df):
    for col in df.columns:
        if col == "ACCOUNT_ID":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def rank_normalize(x):
    return pd.Series(np.asarray(x)).rank(pct=True, method="average").values.astype(np.float32)


def timer(label):
    return _Timer(label)


class _Timer:
    def __init__(self, label):
        self.label = label
        self.start = None

    def __enter__(self):
        self.start = time.time()
        print(f"{self.label} started")
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.time() - self.start
        print(f"{self.label} completed in {elapsed:.2f}s | memory {memory_mb():.1f} MB")
        gc.collect()
