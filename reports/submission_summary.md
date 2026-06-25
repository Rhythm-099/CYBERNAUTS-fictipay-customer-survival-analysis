# Submission Output Summary

## File Overview

| Item | Value |
|---|---:|
| Output file | `AIUB_Turtlers.csv` |
| Total prediction rows | 255,000 |
| Total columns | 5 |
| Account identifier | `ACCOUNT_ID` |
| Prediction target format | Survival-risk submission |

## Submission Columns

| Column | Description |
|---|---|
| `ACCOUNT_ID` | Customer account identifier |
| `RISK_SCORE` | Customer churn-risk ranking score; higher value indicates higher predicted churn risk |
| `SURV_PROB_30D` | Predicted probability of remaining active beyond 30 days |
| `SURV_PROB_60D` | Predicted probability of remaining active beyond 60 days |
| `SURV_PROB_90D` | Predicted probability of remaining active beyond 90 days |

## Validation Checks

| Check | Result |
|---|---:|
| Required column order matched | Passed |
| Missing values | 0 |
| Duplicate `ACCOUNT_ID` values | 0 |
| Non-negative `RISK_SCORE` | Passed |
| Survival probabilities within 0 to 1 | Passed |
| `SURV_PROB_30D >= SURV_PROB_60D` violations | 0 |
| `SURV_PROB_60D >= SURV_PROB_90D` violations | 0 |

## Prediction Statistics

| Metric | Minimum | Maximum | Mean | Median | Standard Deviation |
|---|---:|---:|---:|---:|---:|
| `RISK_SCORE` | 0.000287 | 0.999986 | 0.500002 | 0.499415 | 0.288325 |
| `SURV_PROB_30D` | 0.852851 | 0.999991 | 0.999553 | 0.999741 | 0.001821 |
| `SURV_PROB_60D` | 0.582191 | 0.999941 | 0.997152 | 0.998290 | 0.006127 |
| `SURV_PROB_90D` | 0.001064 | 0.998940 | 0.952665 | 0.969602 | 0.049696 |

## Quartile Summary

| Metric | 25th Percentile | 50th Percentile | 75th Percentile |
|---|---:|---:|---:|
| `RISK_SCORE` | 0.250237 | 0.499415 | 0.749823 |
| `SURV_PROB_30D` | 0.999428 | 0.999741 | 0.999885 |
| `SURV_PROB_60D` | 0.996218 | 0.998290 | 0.999238 |
| `SURV_PROB_90D` | 0.934101 | 0.969602 | 0.986332 |

## Interpretation

The submission contains one prediction row per customer account. The risk score distribution spans nearly the full 0 to 1 range, providing a customer-level ranking signal for churn prioritization. The 30-day and 60-day survival probabilities are concentrated near 1.0, indicating high short-term predicted survival for most accounts. The 90-day survival probability shows wider separation, reflecting stronger medium-term churn differentiation.

The validation checks confirm that the output satisfies the required five-column format, contains no missing or duplicate account entries, keeps risk scores non-negative, keeps survival probabilities within the valid probability range, and preserves the required monotonic survival ordering across 30, 60, and 90 days.
