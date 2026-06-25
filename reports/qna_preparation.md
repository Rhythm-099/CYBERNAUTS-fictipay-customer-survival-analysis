# Presentation Q&A Preparation

## Why was survival analysis used instead of binary classification?

The task required churn timing, not only churn status. Survival analysis can model duration until churn and can handle right-censored customers correctly.

## What does right-censoring mean in this problem?

A right-censored customer did not churn within the observed study period. This does not prove the customer will never churn; it only means the event was not observed before the study ended.

## Why is `RISK_SCORE` separate from survival probabilities?

`RISK_SCORE` is used for ranking customers by churn risk. The survival probability columns estimate the probability of remaining active beyond fixed time horizons. Ranking and calibration are related but not identical objectives.

## Why did the pipeline use multiple models?

Different survival models capture different signals. Cox-style models focus on hazard ranking, AFT models directly estimate time-to-event behaviour, and tree-based models capture nonlinear feature interactions.

## Why was rank normalization used for risk scores?

The concordance index evaluates ordering. Rank normalization puts model outputs on a comparable scale before blending, preventing one model from dominating only because of score magnitude.

## How were survival probabilities kept valid?

After blending, the probabilities were constrained so that 30-day survival is greater than or equal to 60-day survival, and 60-day survival is greater than or equal to 90-day survival.

## What were the most important feature categories?

The most relevant feature groups were recent transaction inactivity, transaction frequency, monetary activity, transaction type behaviour, balance behaviour, and account tenure.

## How can the business use the model?

The business can prioritize high-risk customers for immediate retention actions, target medium-risk customers with engagement campaigns, and avoid unnecessary discounts for low-risk customers.

## What is the main limitation?

The model is built for the specific competition definition of churn and depends on the available transaction, KYC, and balance data. Production use would require ongoing monitoring, validation, and controlled retention experiments.

## How would the solution be improved further?

Possible improvements include deeper time-window features, customer sequence modelling, stronger calibration, uplift modelling for campaign response, and periodic retraining with new activity data.
