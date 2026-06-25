# Project Summary

## Title

CYBERNAUTS FictiPay Customer Survival Analysis

## Achievement

2nd Runner-Up, NSUCEC CYBERNAUTS DATATHON 2026.

## Problem

The challenge required customer lifetime modelling for a financial transaction platform. The objective was to predict customer churn timing from an observation cutoff date rather than producing only a binary churn label.

The final output required one prediction row per customer containing a churn risk score and survival probabilities at 30, 60, and 90 days.

## Data Inputs

The modelling pipeline was built from three customer-level evidence sources:

- Transaction history
- Customer KYC and account profile data
- Day-end balance records

Raw competition data is not included in this repository.

## Output Schema

| Column | Description |
|---|---|
| `ACCOUNT_ID` | Customer account identifier |
| `RISK_SCORE` | Relative churn risk, where higher values indicate higher risk |
| `SURV_PROB_30D` | Probability of remaining active beyond 30 days |
| `SURV_PROB_60D` | Probability of remaining active beyond 60 days |
| `SURV_PROB_90D` | Probability of remaining active beyond 90 days |

## Validation Rules

The final output was validated against the following constraints:

- Required columns present
- One row per test account
- No missing values
- Non-negative risk scores
- Survival probabilities within the range 0 to 1
- Monotonic survival probabilities: `SURV_PROB_30D >= SURV_PROB_60D >= SURV_PROB_90D`

## Result Summary

| Metric | Value |
|---|---:|
| Prediction rows | 255,000 |
| Missing values | 0 |
| Duplicate account IDs | 0 |
| Minimum risk score | 0.000287 |
| Maximum risk score | 0.999986 |
| Mean risk score | 0.500002 |
| Mean 30-day survival probability | 0.999553 |
| Mean 60-day survival probability | 0.997152 |
| Mean 90-day survival probability | 0.952665 |

## Business Value

The model supports retention planning by identifying customers with high near-term churn risk. Instead of applying the same campaign to every customer, risk and survival estimates can be used to prioritize intervention timing, campaign budget, and customer engagement strategy.
