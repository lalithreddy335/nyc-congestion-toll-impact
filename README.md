# 🚕 Did NYC's Congestion Toll Work? ML + Causal Analysis of Rideshare & Taxi Trips

## Business Problem
Since January 5, 2025, vehicles entering Manhattan at or below 60th Street pay a
congestion toll. Uber and Lyft trips in the zone pay $1.50 each; taxis pay $0.75.
The policy aims to cut traffic and raise transit funding, but its real effect on
rideshare demand, fares, driver earnings and traffic speed is still debated.

This project uses NYC TLC trip records, hundreds of millions of trips from
2023–2025, to measure what the toll actually changed, test whether the toll caused
it, and recommend how a rideshare platform should respond.

## Stakeholder
Director of Marketplace Strategy (NYC) at a rideshare company

## Key Business Questions
1. How many trips did the toll change vs an ML-predicted "no toll" baseline?
2. Is the change causal? (difference-in-differences)
3. Who bore the cost: riders, drivers or platforms?
4. Did traffic get faster? (trip speed, pickup wait times)
5. Which neighborhoods won or lost? (clustering)
6. What drives demand, and where should drivers be positioned now?

## Trip Groups
| Group | Definition | Role |
|---|---|---|
| Inside | Pickup and dropoff both in the zone | Treated |
| Crossing | One end in the zone | Treated |
| Outside | Neither end in the zone | Control |

## KPIs
| KPI | Definition |
|---|---|
| Trip volume | Trips per day, by trip group |
| Toll effect | Actual trips − ML-predicted trips |
| Average fare | Rider cost per trip, including the toll |
| Driver pay per hour | Driver pay ÷ trip hours |
| Trip speed | Miles ÷ trip time (congestion proxy) |
| Pickup wait | Minutes from request to driver arrival |
| Toll revenue | Sum of `cbd_congestion_fee` |

## Methods
- XGBoost forecasting to build the no-toll counterfactual
- Difference-in-differences (inside/crossing vs outside)
- K-means clustering to segment neighborhoods by impact
- SHAP to explain what drives demand

## Data
NYC TLC Trip Record Data (High Volume For-Hire Vehicles + Yellow Taxi),
January 2023 to the latest available month. Each month is processed with DuckDB
into an hourly zone-to-zone summary table (sums and counts), which keeps the full
history in a few hundred MB.

## Tools
Python (pandas, scikit-learn, XGBoost, SHAP, statsmodels) · DuckDB · Tableau

## Key Findings
*Coming soon*
