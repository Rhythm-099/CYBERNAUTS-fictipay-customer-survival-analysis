# Submission Output Summary

The final pipeline generated a valid survival prediction file for 255,000 customer accounts.

## Schema

| Column | Description |
|---|---|
| `ACCOUNT_ID` | Customer account identifier |
| `RISK_SCORE` | Churn risk ranking score |
| `SURV_PROB_30D` | Survival probability beyond 30 days |
| `SURV_PROB_60D` | Survival probability beyond 60 days |
| `SURV_PROB_90D` | Survival probability beyond 90 days |

## Validation Checks

| Check | Status |
|---|---|
| Required columns present | Passed |
| Missing values | 0 |
| Duplicate account IDs | 0 |
| Non-negative risk score | Passed |
| Probability range 0 to 1 | Passed |
| 30-day survival >= 60-day survival | Passed |
| 60-day survival >= 90-day survival | Passed |

## Prediction Statistics

| Metric | Value |
|---|---:|
| Rows | 255,000 |
| Risk score minimum | 0.000287 |
| Risk score maximum | 0.999986 |
| Risk score mean | 0.500002 |
| Risk score standard deviation | 0.288325 |
| 30-day survival mean | 0.999553 |
| 60-day survival mean | 0.997152 |
| 90-day survival mean | 0.952665 |
| 30-day survival minimum | 0.852851 |
| 60-day survival minimum | 0.582191 |
| 90-day survival minimum | 0.001064 |

## Interpretation

Short-term survival probabilities were generally high at 30 and 60 days, while 90-day survival showed stronger separation across customers. This pattern is consistent with cumulative churn risk increasing over time.
