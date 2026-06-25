import gc
import os
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from scipy.stats import norm
from lifelines.utils import concordance_index
from config import DATA_DIR, WORK_DIR, FEATURES_PATH, PREDICTIONS_PATH, RANDOM_SEED, N_SPLITS, TRAIN_CHUNK_SIZE, USE_GPU, HORIZONS
from utils import resolve_data_dir, ensure_dir, reduce_memory, clean_numeric, rank_normalize, memory_mb, timer


def prepare_data(data_dir):
    train_labels = pd.read_csv(data_dir / "train_labels.csv")
    test_ids = pd.read_csv(data_dir / "test.csv")
    features = pd.read_parquet(FEATURES_PATH)
    train = train_labels.merge(features, on="ACCOUNT_ID", how="inner")
    test = test_ids.merge(features, on="ACCOUNT_ID", how="inner")
    feature_cols = [c for c in features.columns if c != "ACCOUNT_ID"]
    for col in feature_cols:
        train[col] = pd.to_numeric(train[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
        test[col] = pd.to_numeric(test[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    y = train[["DURATION_DAYS", "EVENT_FLAG"]].copy()
    for h in HORIZONS:
        y[f"y_{h}"] = (y["DURATION_DAYS"] > h).astype("int8")
    y["y_lower"] = y["DURATION_DAYS"].clip(lower=1e-5).astype("float32")
    y["y_upper"] = np.where(y["EVENT_FLAG"] == 1, y["DURATION_DAYS"].clip(lower=1e-5), np.inf).astype("float32")
    x_train = train[feature_cols].astype("float32").values
    x_test = test[feature_cols].astype("float32").values
    test_account_ids = test["ACCOUNT_ID"].values
    del features, train, test
    gc.collect()
    return x_train, x_test, y, test_account_ids, feature_cols


def device_params():
    if USE_GPU:
        return {"tree_method": "hist", "device": "cuda"}
    return {"tree_method": "hist"}


def train_xgb_chunks(params, dtrain, dvalid=None, num_rounds=500, chunk_size=20, early_stopping_rounds=80, eval_name="valid"):
    booster = None
    best_score = None
    best_iter = 0
    rounds_without_improvement = 0
    evals = []
    if dvalid is not None:
        evals = [(dvalid, eval_name)]
    trained = 0
    while trained < num_rounds:
        step = min(chunk_size, num_rounds - trained)
        evals_result = {}
        booster = xgb.train(params, dtrain, num_boost_round=step, evals=evals, xgb_model=booster, verbose_eval=False, evals_result=evals_result)
        trained += step
        if dvalid is not None and eval_name in evals_result:
            metric_name = list(evals_result[eval_name].keys())[0]
            score = evals_result[eval_name][metric_name][-1]
            if best_score is None or score < best_score:
                best_score = score
                best_iter = trained
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += step
            if trained == step or trained % 100 == 0 or rounds_without_improvement >= early_stopping_rounds:
                print(f"round {trained} | {eval_name}-{metric_name}: {score:.6f} | best {best_score:.6f} at {best_iter}")
            if rounds_without_improvement >= early_stopping_rounds:
                break
    return booster


def cox_labels(y):
    duration = y["DURATION_DAYS"].clip(lower=1e-5).values.astype("float32")
    event = y["EVENT_FLAG"].values.astype(bool)
    return np.where(event, duration, -duration).astype("float32")


def baseline_cumulative_hazard(durations, events, hazards):
    durations = np.asarray(durations, dtype="float64")
    events = np.asarray(events, dtype="int32")
    hazards = np.asarray(hazards, dtype="float64")
    hazards = np.clip(hazards, 1e-8, np.percentile(hazards, 99.9))
    order = np.argsort(durations)
    d = durations[order]
    e = events[order]
    h = hazards[order]
    risk_suffix = np.cumsum(h[::-1])[::-1]
    event_times = np.sort(np.unique(d[e == 1]))
    out_times = []
    out_haz = []
    ch = 0.0
    for t in event_times:
        idx = np.searchsorted(d, t, side="left")
        risk_sum = risk_suffix[idx] if idx < len(risk_suffix) else 0.0
        event_count = np.sum((d == t) & (e == 1))
        if risk_sum > 0:
            ch += event_count / risk_sum
        out_times.append(t)
        out_haz.append(ch)
    if len(out_times) == 0:
        return np.array([0.0]), np.array([0.0])
    return np.asarray(out_times), np.asarray(out_haz)


def hazard_at(times, hazards, horizon):
    idx = np.searchsorted(times, horizon, side="right") - 1
    if idx < 0:
        return 0.0
    return float(hazards[idx])


def cox_survival(times, hazards, risk, horizons):
    risk = np.clip(np.asarray(risk, dtype="float64"), 1e-8, np.percentile(risk, 99.9))
    probs = []
    for h in horizons:
        ch = hazard_at(times, hazards, h)
        probs.append(np.exp(-ch * risk))
    return probs


def enforce_survival(p30, p60, p90):
    p30 = np.clip(p30, 0, 1)
    p60 = np.clip(p60, 0, 1)
    p90 = np.clip(p90, 0, 1)
    p60 = np.minimum(p30, p60)
    p90 = np.minimum(p60, p90)
    return p30, p60, p90


def evaluate_fold(duration, event, risk):
    return concordance_index(duration, -risk, event)


def main():
    data_dir = resolve_data_dir(DATA_DIR)
    ensure_dir(WORK_DIR)
    with timer("Data preparation"):
        X_train, X_test, y, test_account_ids, feature_cols = prepare_data(data_dir)
    n = len(y)
    n_test = len(test_account_ids)
    arrays = {}
    for name in ["cox", "rf", "aft", "h30", "h60", "h90"]:
        arrays[f"oof_risk_{name}"] = np.zeros(n, dtype="float32")
        arrays[f"test_risk_{name}"] = np.zeros(n_test, dtype="float32")
        for h in HORIZONS:
            arrays[f"oof_prob_{h}_{name}"] = np.zeros(n, dtype="float32")
            arrays[f"test_prob_{h}_{name}"] = np.zeros(n_test, dtype="float32")
    base_params = device_params()
    cox_params = {
        "objective": "survival:cox",
        "eval_metric": "cox-nloglik",
        "learning_rate": 0.045,
        "max_depth": 6,
        "min_child_weight": 30,
        "subsample": 0.82,
        "colsample_bytree": 0.82,
        "reg_lambda": 2.0,
        "verbosity": 0,
        "seed": RANDOM_SEED,
        **base_params
    }
    rf_params = {
        "objective": "survival:cox",
        "eval_metric": "cox-nloglik",
        "booster": "gbtree",
        "num_parallel_tree": 80,
        "learning_rate": 1.0,
        "max_depth": 9,
        "min_child_weight": 20,
        "subsample": 0.72,
        "colsample_bytree": 0.72,
        "reg_lambda": 1.5,
        "verbosity": 0,
        "seed": RANDOM_SEED,
        **base_params
    }
    aft_params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": 1.20,
        "learning_rate": 0.045,
        "max_depth": 6,
        "min_child_weight": 30,
        "subsample": 0.82,
        "colsample_bytree": 0.82,
        "reg_lambda": 2.0,
        "verbosity": 0,
        "seed": RANDOM_SEED,
        **base_params
    }
    clf_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.04,
        "max_depth": 6,
        "min_child_weight": 30,
        "subsample": 0.82,
        "colsample_bytree": 0.82,
        "reg_lambda": 2.0,
        "verbosity": 0,
        "seed": RANDOM_SEED,
        **base_params
    }
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    overall_start = time.time()
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y["EVENT_FLAG"]), 1):
        fold_start = time.time()
        print(f"Fold {fold}/{N_SPLITS}")
        X_tr = X_train[tr_idx]
        X_va = X_train[va_idx]
        y_tr = y.iloc[tr_idx].reset_index(drop=True)
        y_va = y.iloc[va_idx].reset_index(drop=True)
        dvalid = xgb.DMatrix(X_va)
        dtest = xgb.DMatrix(X_test)
        y_tr_cox = cox_labels(y_tr)
        y_va_cox = cox_labels(y_va)
        dtrain_cox = xgb.DMatrix(X_tr, label=y_tr_cox)
        dvalid_cox = xgb.DMatrix(X_va, label=y_va_cox)
        model_cox = train_xgb_chunks(cox_params, dtrain_cox, dvalid_cox, num_rounds=900, chunk_size=TRAIN_CHUNK_SIZE, early_stopping_rounds=80)
        val_cox = model_cox.predict(dvalid)
        test_cox = model_cox.predict(dtest)
        arrays["oof_risk_cox"][va_idx] = val_cox
        arrays["test_risk_cox"] += test_cox.astype("float32") / N_SPLITS
        train_cox = model_cox.predict(dtrain_cox)
        bt, bh = baseline_cumulative_hazard(y_tr["DURATION_DAYS"].values, y_tr["EVENT_FLAG"].values, train_cox)
        for h, vp, tp in zip(HORIZONS, cox_survival(bt, bh, val_cox, HORIZONS), cox_survival(bt, bh, test_cox, HORIZONS)):
            arrays[f"oof_prob_{h}_cox"][va_idx] = vp
            arrays[f"test_prob_{h}_cox"] += tp.astype("float32") / N_SPLITS
        del model_cox, train_cox
        gc.collect()
        model_rf = xgb.train(rf_params, dtrain_cox, num_boost_round=1)
        val_rf = model_rf.predict(dvalid)
        test_rf = model_rf.predict(dtest)
        arrays["oof_risk_rf"][va_idx] = val_rf
        arrays["test_risk_rf"] += test_rf.astype("float32") / N_SPLITS
        train_rf = model_rf.predict(dtrain_cox)
        rt, rh = baseline_cumulative_hazard(y_tr["DURATION_DAYS"].values, y_tr["EVENT_FLAG"].values, train_rf)
        for h, vp, tp in zip(HORIZONS, cox_survival(rt, rh, val_rf, HORIZONS), cox_survival(rt, rh, test_rf, HORIZONS)):
            arrays[f"oof_prob_{h}_rf"][va_idx] = vp
            arrays[f"test_prob_{h}_rf"] += tp.astype("float32") / N_SPLITS
        del model_rf, train_rf, dtrain_cox, dvalid_cox
        gc.collect()
        dtrain_aft = xgb.DMatrix(X_tr)
        dtrain_aft.set_float_info("label_lower_bound", y_tr["y_lower"].values.astype("float32"))
        dtrain_aft.set_float_info("label_upper_bound", y_tr["y_upper"].values.astype("float32"))
        dvalid_aft = xgb.DMatrix(X_va)
        dvalid_aft.set_float_info("label_lower_bound", y_va["y_lower"].values.astype("float32"))
        dvalid_aft.set_float_info("label_upper_bound", y_va["y_upper"].values.astype("float32"))
        model_aft = train_xgb_chunks(aft_params, dtrain_aft, dvalid_aft, num_rounds=900, chunk_size=TRAIN_CHUNK_SIZE, early_stopping_rounds=80)
        val_logt = model_aft.predict(dvalid)
        test_logt = model_aft.predict(dtest)
        arrays["oof_risk_aft"][va_idx] = -val_logt
        arrays["test_risk_aft"] += (-test_logt).astype("float32") / N_SPLITS
        scale = aft_params["aft_loss_distribution_scale"]
        for h in HORIZONS:
            arrays[f"oof_prob_{h}_aft"][va_idx] = norm.cdf((val_logt - np.log(h)) / scale)
            arrays[f"test_prob_{h}_aft"] += norm.cdf((test_logt - np.log(h)) / scale).astype("float32") / N_SPLITS
        del model_aft, dtrain_aft, dvalid_aft, val_logt, test_logt
        gc.collect()
        for h in HORIZONS:
            label = (y_tr["DURATION_DAYS"].values > h).astype("int8")
            label_va = (y_va["DURATION_DAYS"].values > h).astype("int8")
            dtrain_h = xgb.DMatrix(X_tr, label=label)
            dvalid_h = xgb.DMatrix(X_va, label=label_va)
            model_h = train_xgb_chunks(clf_params, dtrain_h, dvalid_h, num_rounds=800, chunk_size=TRAIN_CHUNK_SIZE, early_stopping_rounds=80)
            val_surv = model_h.predict(dvalid)
            test_surv = model_h.predict(dtest)
            arrays[f"oof_prob_{h}_h{h}"][va_idx] = val_surv
            arrays[f"test_prob_{h}_h{h}"] += test_surv.astype("float32") / N_SPLITS
            arrays[f"oof_risk_h{h}"][va_idx] = 1.0 - val_surv
            arrays[f"test_risk_h{h}"] += (1.0 - test_surv).astype("float32") / N_SPLITS
            del model_h, dtrain_h, dvalid_h
            gc.collect()
        fold_risk = rank_normalize(arrays["oof_risk_cox"][va_idx]) * 0.25 + rank_normalize(arrays["oof_risk_rf"][va_idx]) * 0.20 + rank_normalize(arrays["oof_risk_aft"][va_idx]) * 0.25 + rank_normalize(arrays["oof_risk_h30"][va_idx]) * 0.10 + rank_normalize(arrays["oof_risk_h60"][va_idx]) * 0.10 + rank_normalize(arrays["oof_risk_h90"][va_idx]) * 0.10
        fold_c = evaluate_fold(y_va["DURATION_DAYS"].values, y_va["EVENT_FLAG"].values, fold_risk)
        elapsed = time.time() - fold_start
        print(f"Fold {fold} c-index {fold_c:.6f} | time {elapsed:.2f}s | seconds_per_train_row {elapsed / len(tr_idx):.8f} | memory {memory_mb():.1f} MB")
        del X_tr, X_va, y_tr, y_va, dvalid, dtest
        gc.collect()
    np.savez(PREDICTIONS_PATH, test_account_ids=test_account_ids, feature_count=len(feature_cols), duration=y["DURATION_DAYS"].values, event=y["EVENT_FLAG"].values, **arrays)
    print(f"Saved predictions: {PREDICTIONS_PATH}")
    print(f"Total training time: {time.time() - overall_start:.2f}s")


if __name__ == "__main__":
    main()
