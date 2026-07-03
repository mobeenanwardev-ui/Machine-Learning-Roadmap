# Problem Understanding and Domain Knowledge

## Why this comes before coding

Before touching the data, I first need to understand the actual problem. A model can be technically accurate and still be useless when it answers the wrong question.

A factory may say, “We want to use machine learning for maintenance.” This is too broad because it can mean different things:

| Possible goal | Type of task |
|---|---|
| Predict whether a machine will fail in the next 24 hours | Classification |
| Predict the remaining hours before failure | Regression |
| Detect unusual sensor behaviour | Anomaly detection |
| Group machines with similar behaviour | Clustering |

The goal changes the target, the features, the model, and the evaluation method.

## Turning the real problem into a data problem

A useful problem needs a clear input, output, time frame, and success measure.

| Part | Example definition |
|---|---|
| Business problem | Reduce unexpected machine downtime |
| Prediction question | Will this machine fail within 24 hours? |
| Input | Sensor readings from the previous hour |
| Target | Failure within 24 hours: yes or no |
| Important mistake | Missing a real failure |
| Useful measure | Recall for the failure class |

This is much clearer than simply saying “predict failure.” It tells us what the model should learn and when the prediction must be available.

## Why the cost of mistakes matters

Not every wrong prediction has the same effect.

| Actual situation | Prediction | Practical result |
|---|---|---|
| Failure | Failure | Correct warning |
| Normal | Normal | Correct decision |
| Normal | Failure | Unnecessary inspection |
| Failure | Normal | Real failure is missed |

In a factory, a missed failure may cost much more than an unnecessary inspection. Therefore, accuracy alone may not describe whether the model is useful.

## Why domain knowledge is necessary

The computer does not understand temperature, pressure, customer satisfaction, or failure. It only sees columns and values. Domain knowledge gives the numbers meaning.

| Value | Without domain knowledge | With domain knowledge |
|---|---|---|
| Pressure = 500 | A large value | Impossible for this machine |
| Temperature = 75 °C | A number | Dangerous for this motor type |
| Sensor value = 0 | A low reading | Sensor may be disconnected |
| Repair status = completed | A useful-looking feature | Information created after failure |

Domain knowledge helps identify useful features, impossible values, unreliable labels, missing-data reasons, and information that would not be available at prediction time.

## Example with 500 sensors

Suppose a factory gives me data from 500 sensors. I should not immediately use every column. I first need to ask:

1. What does each sensor measure?
2. Are several sensors measuring almost the same behaviour?
3. How often is each sensor recorded?
4. Are all sensors available on every machine?
5. Does any feature appear only after the failure?
6. Are the failure labels reliable?

A simple check might look like this:

| Feature | Meaning | Available before failure? | Decision |
|---|---|---|---|
| Temperature | Motor heat | Yes | Consider using |
| Vibration | Mechanical movement | Yes | Consider using |
| Repair completed | Repair status | No | Remove from prediction data |
| Machine ID | Identifier | Yes | Check whether it adds real information |

The repair feature could produce high accuracy, but it cannot help predict an event that has already happened.

## Questions to answer before modeling

| Area | Main question |
|---|---|
| Objective | What exactly should be predicted or discovered? |
| User | Who will use the output? |
| Timing | When must the result be available? |
| Data | Which information exists at that time? |
| Errors | Which wrong decision costs more? |
| Success | Which measure shows practical value? |

## Main lesson

> Good machine learning starts with a clear real-world question, not with a model.

A reliable solution must answer the right question using information that is genuinely available when the prediction is made.
