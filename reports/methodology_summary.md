# Methodology Summary

## Pipeline

The solution follows a script-based survival modelling pipeline:

1. Load customer, transaction, and balance data
2. Build account-level behavioural features
3. Prepare survival targets from duration and event status
4. Train survival-oriented models using cross-validation
5. Generate out-of-fold and test predictions
6. Optimize ensemble weights
7. Validate the final submission file

## Feature Engineering

The raw data was transformed into customer-level features. Each row represents one account.

Feature groups included:

- Transaction recency
- Transaction frequency
- Transaction monetary behaviour
- Incoming and outgoing activity
- Transaction-type behaviour
- Account tenure
- Balance level
- Balance volatility
- Zero-balance indicators
- Recent inactivity indicators

The feature strategy was based on the churn definition, where extended inactivity is the main event signal.

## Survival Target Handling

The target consisted of duration and event status.

- `EVENT_FLAG = 1` indicates observed churn
- `EVENT_FLAG = 0` indicates right-censoring

Right-censored customers were not treated as negative churn cases. They were handled using survival modelling methods that preserve time-to-event information.

## Models

The pipeline used an ensemble of XGBoost-based models.

| Model | Role |
|---|---|
| XGBoost Cox PH | Hazard-based churn ranking |
| XGBoost random-forest-style survival model | Nonlinear interaction capture |
| XGBoost AFT | Direct time-to-event modelling with censoring |
| XGBoost horizon classifiers | Fixed-horizon survival probability estimation |

## Ensemble

Risk scores from different models were rank-normalized before blending because the concordance index evaluates ordering rather than absolute score scale.

Survival probabilities were blended separately for the 30-day, 60-day, and 90-day horizons. A monotonicity correction was applied after blending to ensure valid survival curves.

## Validation

The final validation script checks:

- Column order
- Row coverage
- Missing values
- Probability range
- Risk score range
- Survival probability monotonicity

This ensures that the generated file is structurally valid before submission.
