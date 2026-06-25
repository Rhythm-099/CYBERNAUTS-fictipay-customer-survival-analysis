import json
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from lifelines.utils import concordance_index
from config import WORK_DIR, PREDICTIONS_PATH, SUBMISSION_PATH, HORIZONS, RANDOM_SEED
from utils import ensure_dir, rank_normalize, timer


def enforce_survival(p30, p60, p90):
    p30 = np.clip(p30, 0, 1)
    p60 = np.clip(p60, 0, 1)
    p90 = np.clip(p90, 0, 1)
    p60 = np.minimum(p30, p60)
    p90 = np.minimum(p60, p90)
    return p30, p60, p90


def evaluate(duration, event, risk, p30, p60, p90):
    c_idx = concordance_index(duration, -risk, event)
    c_score = max(0, (c_idx - 0.5) / 0.5) * 40
    y30 = (duration > 30).astype(int)
    y60 = (duration > 60).astype(int)
    y90 = (duration > 90).astype(int)
    bs30 = brier_score_loss(y30, p30)
    bs60 = brier_score_loss(y60, p60)
    bs90 = brier_score_loss(y90, p90)
    ibs = (bs30 + bs60 + bs90) / 3.0
    ibs_score = max(0, (0.05 - ibs) / 0.05) * 30
    return c_idx, ibs, c_score + ibs_score


def weighted_sum(items, weights):
    out = np.zeros_like(items[0], dtype="float64")
    for x, w in zip(items, weights):
        out += x * w
    return out


def normalized_weights(n, rng, total=500):
    base = []
    eye = np.eye(n)
    for row in eye:
        base.append(row)
    base.append(np.ones(n) / n)
    for _ in range(total):
        base.append(rng.dirichlet(np.ones(n)))
    return base


def main():
    ensure_dir(WORK_DIR)
    with timer("Load model predictions"):
        data = np.load(PREDICTIONS_PATH, allow_pickle=True)
        duration = data["duration"].astype("float64")
        event = data["event"].astype("int32")
        test_account_ids = data["test_account_ids"]
    risk_models = ["cox", "rf", "aft", "h30", "h60", "h90"]
    prob_models = ["cox", "rf", "aft"]
    oof_risk_items = [rank_normalize(data[f"oof_risk_{m}"]) for m in risk_models]
    test_risk_items = [rank_normalize(data[f"test_risk_{m}"]) for m in risk_models]
    prob_candidates = {}
    for h in HORIZONS:
        prob_candidates[h] = []
        for m in prob_models:
            prob_candidates[h].append(data[f"oof_prob_{h}_{m}"].astype("float64"))
        prob_candidates[h].append(data[f"oof_prob_{h}_h{h}"].astype("float64"))
    test_prob_candidates = {}
    for h in HORIZONS:
        test_prob_candidates[h] = []
        for m in prob_models:
            test_prob_candidates[h].append(data[f"test_prob_{h}_{m}"].astype("float64"))
        test_prob_candidates[h].append(data[f"test_prob_{h}_h{h}"].astype("float64"))
    rng = np.random.default_rng(RANDOM_SEED)
    sample_size = min(120000, len(duration))
    sample_idx = rng.choice(len(duration), size=sample_size, replace=False)
    best = {"score": -1.0, "c_index": 0.0, "ibs": 1.0, "risk_weights": None, "prob_weights": None}
    risk_weight_list = normalized_weights(len(risk_models), rng, total=700)
    prob_weight_list = normalized_weights(4, rng, total=350)
    with timer("Weight optimization"):
        for rw in risk_weight_list:
            risk = weighted_sum([x[sample_idx] for x in oof_risk_items], rw)
            for pw in prob_weight_list:
                p30 = weighted_sum([x[sample_idx] for x in prob_candidates[30]], pw)
                p60 = weighted_sum([x[sample_idx] for x in prob_candidates[60]], pw)
                p90 = weighted_sum([x[sample_idx] for x in prob_candidates[90]], pw)
                p30, p60, p90 = enforce_survival(p30, p60, p90)
                c_idx, ibs, score = evaluate(duration[sample_idx], event[sample_idx], risk, p30, p60, p90)
                if score > best["score"] or (abs(score - best["score"]) < 1e-9 and c_idx > best["c_index"]):
                    best = {"score": float(score), "c_index": float(c_idx), "ibs": float(ibs), "risk_weights": rw.tolist(), "prob_weights": pw.tolist()}
    risk_full = weighted_sum(oof_risk_items, np.asarray(best["risk_weights"]))
    p30_full = weighted_sum(prob_candidates[30], np.asarray(best["prob_weights"]))
    p60_full = weighted_sum(prob_candidates[60], np.asarray(best["prob_weights"]))
    p90_full = weighted_sum(prob_candidates[90], np.asarray(best["prob_weights"]))
    p30_full, p60_full, p90_full = enforce_survival(p30_full, p60_full, p90_full)
    c_idx_full, ibs_full, score_full = evaluate(duration, event, risk_full, p30_full, p60_full, p90_full)
    print(f"OOF c-index: {c_idx_full:.6f}")
    print(f"OOF IBS: {ibs_full:.6f}")
    print(f"OOF score estimate: {score_full:.6f}")
    risk_test = weighted_sum(test_risk_items, np.asarray(best["risk_weights"]))
    p30_test = weighted_sum(test_prob_candidates[30], np.asarray(best["prob_weights"]))
    p60_test = weighted_sum(test_prob_candidates[60], np.asarray(best["prob_weights"]))
    p90_test = weighted_sum(test_prob_candidates[90], np.asarray(best["prob_weights"]))
    p30_test, p60_test, p90_test = enforce_survival(p30_test, p60_test, p90_test)
    risk_test = rank_normalize(risk_test)
    submission = pd.DataFrame({
        "ACCOUNT_ID": test_account_ids,
        "RISK_SCORE": risk_test,
        "SURV_PROB_30D": p30_test,
        "SURV_PROB_60D": p60_test,
        "SURV_PROB_90D": p90_test
    })
    submission.to_csv(SUBMISSION_PATH, index=False)
    metadata = {
        "risk_models": risk_models,
        "prob_models_per_horizon": prob_models + ["horizon_classifier"],
        "risk_weights": best["risk_weights"],
        "prob_weights": best["prob_weights"],
        "oof_c_index": float(c_idx_full),
        "oof_ibs": float(ibs_full),
        "oof_score_estimate": float(score_full),
        "submission_path": str(SUBMISSION_PATH)
    }
    with open(WORK_DIR / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved submission: {SUBMISSION_PATH}")
    print(submission.head())


if __name__ == "__main__":
    main()
