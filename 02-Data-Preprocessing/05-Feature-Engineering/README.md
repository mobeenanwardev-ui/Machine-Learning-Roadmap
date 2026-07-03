# Feature Engineering

## What feature engineering means

Feature engineering means creating, selecting, or transforming input variables so that the data represents the real problem more clearly.

A model only sees the features I give it. If the useful pattern is hidden inside raw columns, a better feature can make it easier to learn.

## Example: fuel efficiency

Suppose a vehicle dataset contains distance and fuel use:

| Vehicle | Distance (km) | Fuel used (litres) |
|---|---:|---:|
| A | 400 | 20 |
| B | 420 | 35 |
| C | 300 | 15 |

The new feature can be calculated as:

`fuel_efficiency = distance / fuel_used`

| Vehicle | Fuel efficiency (km/l) |
|---|---:|
| A | 20 |
| B | 12 |
| C | 20 |

Vehicle B travels the longest distance, but it is less efficient. The new feature describes the behaviour more directly.

## More examples

| Raw data | New feature | Why it may help |
|---|---|---|
| Date of birth | Age | Easier to use directly |
| Timestamp | Hour, weekday, month | Captures time patterns |
| Height and weight | BMI | Represents body proportion |
| Price and quantity | Total order value | Represents spending |
| Sensor history | Moving average | Reduces short-term noise |
| Login records | Logins in last 30 days | Represents recent activity |

## Creation, selection, and transformation

| Process | Meaning | Example |
|---|---|---|
| Feature creation | Build a new variable | Fuel efficiency |
| Feature selection | Keep useful variables | Remove an irrelevant ID |
| Feature transformation | Change representation | Scaling or a logarithm |

These processes are related, but they solve different problems.

## Why domain knowledge matters

Not every mathematical combination is meaningful.

| Combination | Useful? | Reason |
|---|---|---|
| Distance / fuel | Yes | Measures efficiency |
| Total price / item count | Yes | Average item price |
| Customer age / postal code | No | No logical meaning |

A feature should have a sensible interpretation and should be available at prediction time.

## Feature engineering is not the same as correlation

Correlation measures a relationship between existing variables. Feature engineering creates a new representation.

A feature may also be useful even when its simple correlation with the target is low. The relationship may be nonlinear or may only appear when combined with another feature.

## Example with sensor data

Suppose a temperature sensor records:

| Minute | Temperature |
|---|---:|
| 1 | 60 |
| 2 | 61 |
| 3 | 62 |
| 4 | 79 |
| 5 | 81 |

The latest temperature is useful, but a new feature such as the increase during the last five minutes may reveal that the machine is heating rapidly.

Possible new features are:

- average temperature during the last five minutes,
- maximum temperature,
- change from the first to the latest reading,
- number of readings above a safe limit.

## Main lesson

> Feature engineering turns domain understanding into variables that make the useful pattern clearer to the model.

A feature should be created for a logical reason and then tested to confirm that it improves performance on unseen data.
