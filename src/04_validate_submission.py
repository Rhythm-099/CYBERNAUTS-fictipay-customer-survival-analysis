import pandas as pd
from config import DATA_DIR, SUBMISSION_PATH
from utils import resolve_data_dir


def main():
    data_dir = resolve_data_dir(DATA_DIR)
    test = pd.read_csv(data_dir / "test.csv")
    sub = pd.read_csv(SUBMISSION_PATH)
    expected_cols = ["ACCOUNT_ID", "RISK_SCORE", "SURV_PROB_30D", "SURV_PROB_60D", "SURV_PROB_90D"]
    assert list(sub.columns) == expected_cols
    assert len(sub) == len(test)
    assert sub["ACCOUNT_ID"].is_unique
    assert set(sub["ACCOUNT_ID"]) == set(test["ACCOUNT_ID"])
    assert sub[expected_cols[1:]].notna().all().all()
    assert (sub["RISK_SCORE"] >= 0).all()
    for col in ["SURV_PROB_30D", "SURV_PROB_60D", "SURV_PROB_90D"]:
        assert ((sub[col] >= 0) & (sub[col] <= 1)).all()
    assert (sub["SURV_PROB_30D"] >= sub["SURV_PROB_60D"]).all()
    assert (sub["SURV_PROB_60D"] >= sub["SURV_PROB_90D"]).all()
    print("Submission validation passed")
    print(f"Rows: {len(sub)}")
    print(f"Path: {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
