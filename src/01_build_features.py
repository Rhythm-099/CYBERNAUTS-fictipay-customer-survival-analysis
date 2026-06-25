import gc
import os
import numpy as np
import pandas as pd
import dask.dataframe as dd
from config import DATA_DIR, WORK_DIR, FEATURES_PATH, OBSERVATION_END, OBSERVATION_END_EXCLUSIVE, WINDOWS, TOP_TRX_TYPES
from utils import resolve_data_dir, ensure_dir, reduce_memory, clean_numeric, timer, memory_mb


def flatten_columns(df):
    cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            cols.append("_".join([str(x) for x in c if str(x) != ""]))
        else:
            cols.append(str(c))
    df.columns = cols
    return df


def merge_feature(base, new_df):
    if new_df is None or len(new_df) == 0:
        return base
    return base.merge(new_df, on="ACCOUNT_ID", how="left")


def read_trx(data_dir):
    paths = sorted((data_dir / "transactions").glob("*.parquet"))
    return dd.read_parquet([str(p) for p in paths])


def read_balance(data_dir):
    paths = sorted((data_dir / "dayend_balance").glob("*.parquet"))
    return dd.read_parquet([str(p) for p in paths])


def base_population(data_dir):
    train = pd.read_csv(data_dir / "train_labels.csv", usecols=["ACCOUNT_ID"])
    test = pd.read_csv(data_dir / "test.csv", usecols=["ACCOUNT_ID"])
    accounts = pd.concat([train, test], ignore_index=True).drop_duplicates("ACCOUNT_ID")
    accounts["ACCOUNT_ID"] = accounts["ACCOUNT_ID"].astype(str)
    return accounts


def kyc_features(data_dir, accounts):
    kyc = pd.read_parquet(data_dir / "kyc.parquet")
    kyc["ACCOUNT_ID"] = kyc["ACCOUNT_ID"].astype(str)
    kyc = accounts.merge(kyc, on="ACCOUNT_ID", how="left")
    if "ACCOUNT_OPEN_DATE" in kyc.columns:
        open_date = pd.to_datetime(kyc["ACCOUNT_OPEN_DATE"], errors="coerce")
        obs = pd.Timestamp(OBSERVATION_END)
        kyc["tenure_days"] = (obs - open_date).dt.days.clip(lower=0).fillna(0)
        kyc = kyc.drop(columns=["ACCOUNT_OPEN_DATE"])
    cat_cols = [c for c in ["ACCOUNT_TYPE", "GENDER", "REGION"] if c in kyc.columns]
    for c in cat_cols:
        kyc[c] = kyc[c].astype("object").where(kyc[c].notna(), "missing").astype(str)
    kyc = pd.get_dummies(kyc, columns=cat_cols, dummy_na=False)
    for c in kyc.columns:
        if c != "ACCOUNT_ID":
            kyc[c] = pd.to_numeric(kyc[c], errors="coerce").fillna(0)
    return reduce_memory(kyc)


def aggregate_trx_side(trx, customer_ids, account_col, prefix):
    trx = trx[(trx["TRX_DATETIME"] >= "2024-01-01") & (trx["TRX_DATETIME"] < OBSERVATION_END_EXCLUSIVE)]
    trx = trx[trx[account_col].isin(customer_ids)]
    trx = trx.assign(TRX_DATE=trx["TRX_DATETIME"].dt.floor("D"))
    trx = trx.rename(columns={account_col: "ACCOUNT_ID"})
    trx["TRX_AMT"] = trx["TRX_AMT"].astype("float64")
    base = trx.groupby("ACCOUNT_ID").agg({"TRX_AMT": ["count", "sum", "mean", "std", "max"], "TRX_DATE": "nunique"}).compute().reset_index()
    base = flatten_columns(base)
    base = base.rename(columns={
        "TRX_AMT_count": f"{prefix}_txn_count_total",
        "TRX_AMT_sum": f"{prefix}_amt_sum_total",
        "TRX_AMT_mean": f"{prefix}_amt_mean_total",
        "TRX_AMT_std": f"{prefix}_amt_std_total",
        "TRX_AMT_max": f"{prefix}_amt_max_total",
        "TRX_DATE_nunique": f"{prefix}_active_days_total"
    })
    last_date = trx.groupby("ACCOUNT_ID")["TRX_DATETIME"].max().compute().reset_index()
    last_date[f"{prefix}_days_since_last_txn"] = (pd.Timestamp(OBSERVATION_END) - pd.to_datetime(last_date["TRX_DATETIME"])).dt.days.clip(lower=0)
    last_date = last_date.drop(columns=["TRX_DATETIME"])
    out = base.merge(last_date, on="ACCOUNT_ID", how="left")
    for w in WINDOWS:
        start = pd.Timestamp(OBSERVATION_END) - pd.Timedelta(days=w-1)
        wdf = trx[trx["TRX_DATETIME"] >= start]
        agg = wdf.groupby("ACCOUNT_ID").agg({"TRX_AMT": ["count", "sum", "mean"], "TRX_DATE": "nunique"}).compute().reset_index()
        agg = flatten_columns(agg)
        agg = agg.rename(columns={
            "TRX_AMT_count": f"{prefix}_txn_count_{w}d",
            "TRX_AMT_sum": f"{prefix}_amt_sum_{w}d",
            "TRX_AMT_mean": f"{prefix}_amt_mean_{w}d",
            "TRX_DATE_nunique": f"{prefix}_active_days_{w}d"
        })
        out = out.merge(agg, on="ACCOUNT_ID", how="left")
    type_counts = trx["TRX_TYPE"].value_counts().compute().head(TOP_TRX_TYPES)
    top_types = [str(x) for x in type_counts.index.tolist()]
    for tx_type in top_types:
        safe = "".join(ch if ch.isalnum() else "_" for ch in tx_type.lower()).strip("_")
        sdf = trx[trx["TRX_TYPE"].astype(str) == tx_type]
        agg = sdf.groupby("ACCOUNT_ID").agg({"TRX_AMT": ["count", "sum"]}).compute().reset_index()
        agg = flatten_columns(agg)
        agg = agg.rename(columns={"TRX_AMT_count": f"{prefix}_{safe}_count", "TRX_AMT_sum": f"{prefix}_{safe}_sum"})
        out = out.merge(agg, on="ACCOUNT_ID", how="left")
    out = clean_numeric(out.fillna(0))
    return reduce_memory(out)


def balance_features(data_dir, accounts):
    bal = read_balance(data_dir)
    bal = bal[bal["ACCOUNT_ID"].isin(set(accounts["ACCOUNT_ID"].tolist()))]
    bal = bal[(bal["DATE"] >= "2024-01-01") & (bal["DATE"] < OBSERVATION_END_EXCLUSIVE)]
    bal["AVAILABLE_BALANCE"] = bal["AVAILABLE_BALANCE"].astype("float64")
    agg = bal.groupby("ACCOUNT_ID").agg({"AVAILABLE_BALANCE": ["mean", "std", "min", "max", "count"]}).compute().reset_index()
    agg = flatten_columns(agg)
    agg = agg.rename(columns={
        "AVAILABLE_BALANCE_mean": "bal_mean_total",
        "AVAILABLE_BALANCE_std": "bal_std_total",
        "AVAILABLE_BALANCE_min": "bal_min_total",
        "AVAILABLE_BALANCE_max": "bal_max_total",
        "AVAILABLE_BALANCE_count": "bal_days_total"
    })
    latest = bal[bal["DATE"] == OBSERVATION_END].groupby("ACCOUNT_ID")["AVAILABLE_BALANCE"].mean().compute().reset_index()
    latest = latest.rename(columns={"AVAILABLE_BALANCE": "bal_last"})
    out = agg.merge(latest, on="ACCOUNT_ID", how="left")
    for w in WINDOWS:
        start = pd.Timestamp(OBSERVATION_END) - pd.Timedelta(days=w-1)
        wdf = bal[bal["DATE"] >= start]
        wagg = wdf.groupby("ACCOUNT_ID").agg({"AVAILABLE_BALANCE": ["mean", "std", "min", "max", "count"]}).compute().reset_index()
        wagg = flatten_columns(wagg)
        wagg = wagg.rename(columns={
            "AVAILABLE_BALANCE_mean": f"bal_mean_{w}d",
            "AVAILABLE_BALANCE_std": f"bal_std_{w}d",
            "AVAILABLE_BALANCE_min": f"bal_min_{w}d",
            "AVAILABLE_BALANCE_max": f"bal_max_{w}d",
            "AVAILABLE_BALANCE_count": f"bal_days_{w}d"
        })
        zero = wdf[wdf["AVAILABLE_BALANCE"] <= 0].groupby("ACCOUNT_ID").size().compute().reset_index()
        zero = zero.rename(columns={0: f"zero_bal_days_{w}d"})
        wagg = wagg.merge(zero, on="ACCOUNT_ID", how="left")
        out = out.merge(wagg, on="ACCOUNT_ID", how="left")
    out = clean_numeric(out.fillna(0))
    for w in WINDOWS:
        if f"zero_bal_days_{w}d" in out.columns and f"bal_days_{w}d" in out.columns:
            out[f"zero_bal_ratio_{w}d"] = out[f"zero_bal_days_{w}d"] / (out[f"bal_days_{w}d"] + 1e-6)
    return reduce_memory(out)


def interaction_features(features):
    for w in WINDOWS:
        oc = f"out_txn_count_{w}d"
        ic = f"in_txn_count_{w}d"
        os = f"out_amt_sum_{w}d"
        ins = f"in_amt_sum_{w}d"
        if oc in features.columns and ic in features.columns:
            features[f"total_txn_count_{w}d"] = features[oc] + features[ic]
            features[f"in_out_count_ratio_{w}d"] = features[ic] / (features[oc] + 1)
            features[f"no_activity_{w}d"] = (features[f"total_txn_count_{w}d"] == 0).astype("int8")
        if os in features.columns and ins in features.columns:
            features[f"net_flow_{w}d"] = features[ins] - features[os]
            features[f"in_out_amt_ratio_{w}d"] = features[ins] / (features[os] + 1)
    if "out_days_since_last_txn" in features.columns and "in_days_since_last_txn" in features.columns:
        features["days_since_last_any_txn"] = np.minimum(features["out_days_since_last_txn"], features["in_days_since_last_txn"])
    if "total_txn_count_7d" in features.columns and "total_txn_count_30d" in features.columns:
        features["activity_rate_7_vs_30"] = (features["total_txn_count_7d"] + 1) / (features["total_txn_count_30d"] + 1)
    if "total_txn_count_30d" in features.columns and "total_txn_count_90d" in features.columns:
        features["activity_rate_30_vs_90"] = (features["total_txn_count_30d"] + 1) / (features["total_txn_count_90d"] + 1)
    if "bal_last" in features.columns and "bal_mean_30d" in features.columns:
        features["bal_last_vs_30d_mean"] = features["bal_last"] / (features["bal_mean_30d"] + 1)
    return features


def main():
    data_dir = resolve_data_dir(DATA_DIR)
    ensure_dir(WORK_DIR)
    print(f"Data directory: {data_dir}")
    print(f"Work directory: {WORK_DIR}")
    with timer("Base population"):
        accounts = base_population(data_dir)
        customer_ids = set(accounts["ACCOUNT_ID"].tolist())
    with timer("KYC features"):
        features = kyc_features(data_dir, accounts)
    with timer("Transaction features"):
        trx = read_trx(data_dir)
        trx["SRC_ACCOUNT"] = trx["SRC_ACCOUNT"].astype(str)
        trx["DST_ACCOUNT"] = trx["DST_ACCOUNT"].astype(str)
        trx["TRX_DATETIME"] = dd.to_datetime(trx["TRX_DATETIME"], errors="coerce")
        out_feats = aggregate_trx_side(trx, customer_ids, "SRC_ACCOUNT", "out")
        in_feats = aggregate_trx_side(trx, customer_ids, "DST_ACCOUNT", "in")
        features = merge_feature(features, out_feats)
        features = merge_feature(features, in_feats)
        del trx, out_feats, in_feats
        gc.collect()
    with timer("Balance features"):
        bal_feats = balance_features(data_dir, accounts)
        features = merge_feature(features, bal_feats)
        del bal_feats
        gc.collect()
    with timer("Final feature processing"):
        features = interaction_features(features.fillna(0))
        for col in features.columns:
            if col != "ACCOUNT_ID":
                features[col] = pd.to_numeric(features[col], errors="coerce").replace([np.inf, -np.inf], 0).fillna(0)
        features = reduce_memory(features)
        features.to_parquet(FEATURES_PATH, index=False)
    print(f"Saved features: {FEATURES_PATH}")
    print(f"Feature shape: {features.shape}")
    print(f"Memory: {memory_mb():.1f} MB")


if __name__ == "__main__":
    main()
