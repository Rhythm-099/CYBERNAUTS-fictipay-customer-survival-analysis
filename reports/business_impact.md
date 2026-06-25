# Business Impact and Retention Strategy

## Objective

The model supports customer retention by estimating both churn risk and churn timing. The output can be used to decide which customers need immediate intervention and which customers can be handled through lighter engagement.

## Risk Segmentation

| Segment | Model Signal | Business Action |
|---|---|---|
| Urgent risk | High risk score and low 30-day survival probability | Immediate retention offer, fee waiver, cashback, or direct outreach |
| Medium-term risk | Moderate 30-day survival with declining 60-day or 90-day survival | Targeted engagement campaign, payment reminders, or merchant-specific offers |
| Low risk | High survival probability across all horizons | Maintain standard communication and avoid unnecessary discount spending |

## Recommended Campaign Design

### Immediate Retention

Target the highest-risk customers with short-term incentives. This group should receive offers that encourage a transaction before the inactivity period becomes permanent churn.

### Behaviour-Based Offers

Campaigns should be matched to the customer's past usage pattern:

- Merchant users: merchant cashback or partner offers
- Bill payment users: bill-payment reminders or utility rewards
- Cash-in inactive users: cash-in fee waiver or bonus
- P2P inactive users: transfer incentives

### Monitoring Cadence

Risk scores should be refreshed regularly as new transaction and balance data becomes available. This turns the model from a one-time prediction file into an operational retention monitoring system.

### Measurement

Retention actions should be evaluated through controlled experiments. A high-risk group can be split into treatment and control groups to measure uplift in activity, reduction in churn, and return on incentive cost.

## Expected Value

The model helps reduce wasted campaign budget by avoiding unnecessary incentives for low-risk customers. It also improves intervention timing by identifying customers who are likely to churn soon rather than only customers who are generally risky.
