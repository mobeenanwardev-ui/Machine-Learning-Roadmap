# Problem Understanding and Domain Knowledge

## What I understood

Before touching the data, I first need to understand the actual problem. This sounds obvious, but it is easy to skip. A model can be technically accurate and still be useless if it solves the wrong problem.

For example, a factory may say, "We want to use machine learning for maintenance." That statement is too broad. I would need to ask:

- Do we want to predict whether a machine will fail?
- Do we want to predict how many hours are left before failure?
- Do we want to detect abnormal sensor behaviour?
- How early must the warning come?
- What is more expensive: a false alarm or a missed failure?

These questions change the whole project. They affect the target variable, the data we need, the model type, and the evaluation metric.

## Why domain knowledge is necessary

The computer does not understand what temperature, pressure, customer satisfaction, or machine failure means. It only sees columns and values.

Domain knowledge gives those values meaning. A manufacturing engineer may know that a pressure reading is impossible, while a data analyst may only see it as a large number. A doctor may know that two measurements should be interpreted together. A bank expert may know that a certain variable cannot legally be used in a decision.

This means domain knowledge helps with:

- choosing useful features,
- identifying impossible values,
- understanding missing data,
- defining the correct target,
- avoiding misleading conclusions,
- and deciding whether the model output is practically useful.

## Example: predictive maintenance

Suppose I have data from 500 sensors and a column called `machine_failed`.

I should not immediately use all 500 sensors. First, I would try to understand:

- what each sensor measures,
- how often it records data,
- whether some sensors are duplicates,
- whether any sensor is only activated after a failure,
- and whether the failure label is reliable.

A sensor that turns on only after the machine has already failed could give excellent model accuracy, but it would be useless for prediction. This would also create data leakage.

## Converting the business problem into a data problem

A useful machine-learning problem should have a clear input and output.

Example:

- **Business problem:** Reduce unexpected machine downtime.
- **Data problem:** Predict whether a machine will fail within the next 24 hours using sensor readings from the previous hour.
- **Input features:** temperature, vibration, pressure, motor current, machine age.
- **Target:** failure within 24 hours: yes or no.
- **Success measure:** high recall, because missing a real failure may be very costly.

This step makes the project measurable.

## Main lesson

> Good machine learning starts with a clear real-world question, not with a model.
