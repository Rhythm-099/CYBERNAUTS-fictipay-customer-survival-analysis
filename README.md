# FictiPay Customer Survival Analysis

A survival-analysis pipeline for customer lifetime prediction, built on transaction history, KYC data, and day-end balance behavior.

This is the AIUB_Turtlers solution for the FictiPay Datathon's advanced survival modelling challenge. The task wasn't just predicting whether a customer churns. It was predicting when, and how their survival probability moves over time.

---

## Project Overview

Most churn problems are framed as binary classification: churn or don't churn. This competition asked for more. Customer lifetime had to be modelled directly, using survival analysis, and the model had to output:

- A customer-level churn risk score
- 30-day survival probability
- 60-day survival probability
- 90-day survival probability

That's a step closer to how an actual retention team would want the problem framed. Knowing which customers are risky matters, but knowing how soon matters just as much.

---

## Competition Problem

The final round centered on customer lifetime and survival analysis for a mobile financial service.

Data provided:

- Transaction records, January to March 2024
- Customer KYC and account information
- Daily end-of-day account balance records
- Survival labels with duration and event status

Churn was defined as 30 consecutive days without a transaction. Customers still active at the end of the study window were right-censored.

The prediction window starts from the observation cutoff date, March 31, 2024. Every customer's survival duration is measured from that point.

Required submission format:

```text
ACCOUNT_ID,RISK_SCORE,SURV_PROB_30D,SURV_PROB_60D,SURV_PROB_90D
```

- `ACCOUNT_ID` identifies the customer.
- `RISK_SCORE` is the churn risk score. Higher means higher expected risk.
- `SURV_PROB_30D` is the probability the customer is still active past 30 days.
- `SURV_PROB_60D` is the same for 60 days.
- `SURV_PROB_90D` is the same for 90 days.

Survival probabilities must follow the monotonic constraint:

```text
SURV_PROB_30D >= SURV_PROB_60D >= SURV_PROB_90D
```

---

## Objective

The pipeline needed to:

1. Turn raw financial activity data into customer-level behavioral features.
2. Handle right-censored churn observations correctly.
3. Predict churn risk with survival-oriented models.
4. Estimate survival probability at 30, 60, and 90 days.
5. Produce a valid submission file.
6. Turn model output into retention recommendations a business team could actually use.

---

## Why Survival Analysis

A classification model tells you if a customer will churn. It says nothing about when. This competition needed the when.

Survival analysis fits because it models event timing directly, handles censored customers without throwing away information, and produces both a risk ranking and a probability estimate, which is more than a churn label gives you.

One distinction matters here: a censored customer isn't a loyal customer. It just means they hadn't churned by the time the observation window closed. Treating "censored" as "safe" would be a mistake, and survival modelling is built specifically to avoid that mistake.

---

## Data Sources

### 1. Transaction Data

Customer activity records: transaction date and time, source account, destination account, transaction type, transaction amount.

Used to build recency, frequency, monetary, activity-decline, and transaction-type features, split by incoming and outgoing flow.

### 2. KYC Data

Account ID, account type, account opening date, gender, region.

Used for customer profile and tenure features.

### 3. Day-End Balance Data

Daily available balance per account.

Used for average balance, last available balance, balance volatility, min/max balance, zero-balance flags, and balance trend.

---

## Feature Engineering

Raw data is collapsed into a single account-level feature table, one row per customer.

### Transaction Recency Features

How recently a customer transacted: days since last transaction, days since last outgoing transaction, days since last incoming transaction, recent inactivity flags.

Recency carries the most weight here, since the churn definition itself is just transaction silence measured in days.

### Transaction Frequency Features

How often a customer transacts: total transaction count, number of active transaction days, transaction count in recent windows, average transactions per active day.

This is what separates engaged customers from customers who are quietly drifting off.

### Monetary Features

Transaction value: total amount, average amount, median amount, max amount, incoming and outgoing value split.

Useful for telling occasional, low-value users apart from customers who actually move money through the platform regularly.

### Transaction-Type Features

P2P activity, merchant payment activity, bill payment activity, cash-in activity, cash-out activity, and the ratios between them.

This is roughly a proxy for what kind of relationship the customer has with the platform, not just how active they are.

### Balance Behavior Features

Average available balance, last available balance, balance standard deviation, zero-balance frequency, balance range, balance trend.

Low balance and low activity aren't the same signal, but they tend to show up together in customers who are about to churn.

### Tenure and Profile Features

Account age, account type, region, gender, derived from KYC data.

New accounts and long-tenure accounts don't churn the same way, so tenure earns its own feature group.

---

## Modelling Approach

The final model is an ensemble of four survival-oriented components, each covering a different angle of the problem.

**XGBoost Cox Proportional Hazards Model.** Estimates relative churn hazard, which feeds the risk ranking. This matters directly because the competition scores `RISK_SCORE` with the concordance index.

**XGBoost Random-Forest-Style Survival Model.** Built with XGBoost tree settings to pick up nonlinear interactions. Low balance by itself might not say much, but low balance paired with recent inactivity is a much stronger churn signal than either feature alone.

**XGBoost Accelerated Failure Time Model.** Models time-to-event directly, and handles censoring through lower and upper label bounds. Churned customers get the observed duration as the event time. Censored customers get an upper bound of infinity.

**XGBoost Horizon Classifiers.** Three separate models, one per horizon (30, 60, 90 days), built specifically to feed the probability columns the submission requires.

---

## Cross-Validation Strategy

Training uses stratified K-fold cross-validation, stratified on event status, so every fold gets a reasonable mix of churned and censored customers.

Out-of-fold predictions are generated for the risk score and all three survival probabilities. This is what makes model blending possible without leaking validation data into training.

---

## Ensemble Strategy

The final prediction blends multiple models.

Risk scores get rank-normalized before blending, since the concordance index only cares about ranking, not the raw scale of the score. Survival probabilities are blended separately at each horizon.

After blending, monotonicity is enforced:

```text
SURV_PROB_30D >= SURV_PROB_60D >= SURV_PROB_90D
```

Without this step, blended outputs could end up logically inconsistent, where a customer's 60-day survival probability comes out higher than their 30-day one.

---

## Evaluation Metrics

**Concordance Index.** Checks whether customers who churn sooner get assigned higher risk scores. Higher c-index, better ranking.

**Brier Score / Integrated Brier Score.** Checks whether the predicted survival probabilities at 30, 60, and 90 days actually match what happened.

One measures ranking quality, the other measures calibration. The competition cares about both.

---

## Main Challenges

**Data scale.** The raw transaction and balance tables were too large to train on directly without running into memory limits, so the pipeline aggregates to account-level features instead of working row by row on raw transactions.

**Defining churn.** A 30-day silence window sounds simple until you have to translate it into recency, duration, and survival time consistently across the pipeline.

**Censoring.** Customers who hadn't churned by the end of the study period can't just be labeled "not churned" and dropped into a normal classifier. Survival models exist for exactly this case.

**Risk and probability aren't the same problem.** A model can rank customers well and still produce poorly calibrated probabilities, or vice versa. That's why risk scoring and probability estimation are handled as separate but connected components rather than one model doing both jobs.

**The monotonic constraint.** Survival probability has to decrease or stay flat over time, never increase. The validation script checks and enforces this before anything gets submitted.

**Runtime and memory.** Training had to run on the hardware actually available, so the code leans on memory-efficient data types, configurable fold counts, GPU support where available, and chunk-based training.

---

## Repository Structure

```text
fictipay-customer-survival-analysis/
│
├── README.md
├── requirements.txt
├── .gitignore
├── run_all.py
│
├── src/
│   ├── config.py
│   ├── utils.py
│   ├── 01_build_features.py
│   ├── 02_train_models.py
│   ├── 03_optimize_and_submit.py
│   └── 04_validate_submission.py
│
├── reports/
│   └── AIUB_Turtlers_Survival_Report.md
│
├── presentation/
│   └── AIUB_Turtlers_Survival_Presentation.pptx
│
├── images/
│   ├── pipeline_diagram.png
│   ├── survival_output_example.png
│   └── leaderboard_or_result_blurred.png
│
└── outputs/
    └── .gitkeep
```

---

## Script Descriptions

| File                            | Purpose                                                                                   |
| ------------------------------- | ----------------------------------------------------------------------------------------- |
| `src/config.py`                 | Path configuration, fold settings, GPU option, training parameters             |
| `src/utils.py`                  | Shared helper functions used across the pipeline                                 |
| `src/01_build_features.py`      | Builds account-level features from KYC, transaction, and balance data                     |
| `src/02_train_models.py`        | Trains XGBoost Cox, XGBoost random-forest-style survival, XGBoost AFT, and horizon models |
| `src/03_optimize_and_submit.py` | Optimizes ensemble weights and writes the final submission file                           |
| `src/04_validate_submission.py` | Validates column order, monotonic survival probabilities, value ranges, and row coverage  |
| `run_all.py`                    | Runs the complete pipeline in order                                                       |

---

## Output

Five columns, in this order:

```text
ACCOUNT_ID
RISK_SCORE
SURV_PROB_30D
SURV_PROB_60D
SURV_PROB_90D
```

The validation script checks for required columns, correct column order, no missing values, non-negative risk scores, probabilities bounded between 0 and 1, monotonic probabilities across horizons, and row count matching the test account list.

---

## Business Use Case

Instead of sending every customer the same retention campaign, risk score and survival probability can be used to decide who needs attention now versus who can wait.

| Customer Segment | Model Signal                                              | Recommended Action                                                  |
| ---------------- | --------------------------------------------------------- | --------------------------------------------------------------------- |
| Urgent risk      | High risk score, low 30-day survival probability          | Immediate retention offer, cashback, fee waiver, or direct outreach |
| Medium-term risk | Moderate 30-day survival, declining 60/90-day survival     | Engagement campaign, merchant offers, bill-payment reminders        |
| Low risk         | High survival probability across all horizons              | Normal communication, no need to spend on discounts                |

Targeting the right customers at the right time keeps retention cost down and the campaign actually useful, rather than blasting offers at everyone.

---

## Reproducibility Notes

The raw competition dataset isn't included here due to data-sharing restrictions.

To reproduce:

1. Place the competition data in the configured data directory.
2. Update paths or environment variables in `src/config.py`.
3. Install dependencies from `requirements.txt`.
4. Run `python run_all.py`.
5. Check the generated output and validation logs.

---

## Limitations

- No raw competition data included.
- Feature generation depends on the original transaction, KYC, and balance files matching the expected structure.
- The model is built around this competition's specific survival target definition, not a general-purpose churn target.
- Survival probabilities only cover the required 30, 60, and 90-day horizons.
- Business recommendations need A/B testing before going into production. None of this has been validated outside the competition setting.

---

## Team

AIUB_Turtlers

Built for the FictiPay Datathon survival-analysis challenge.
