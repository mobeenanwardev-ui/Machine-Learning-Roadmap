# Feature Engineering

## What I understood

Feature engineering means creating, selecting, or transforming input variables so that the data represents the real problem more clearly.

The model only sees the features I give it. If the useful pattern is hidden, a good feature can make the pattern easier to learn.

For example, suppose a dataset contains `distance_travelled` and `fuel_used`. These two columns are useful, but a new feature such as `fuel_efficiency = distance_travelled / fuel_used` may describe the behaviour more directly.

Other examples include:

- extracting hour, weekday, or month from a timestamp,
- calculating age from date of birth,
- combining height and weight into BMI,
- calculating the average of several related sensor readings,
- or converting raw transaction history into number of purchases in the last 30 days.

## Feature engineering is not the same as correlation

Correlation only measures a numerical relationship between existing variables. Feature engineering uses human understanding to create a new representation.

A feature can also be useful even when its simple correlation with the target is low. The relationship may be nonlinear or may only appear when combined with another feature.

## Domain knowledge matters

A model does not know why a feature should be meaningful. A human must decide whether a transformation makes sense.

For example, dividing two columns just because both contain numbers can create meaningless values. A feature should have a logical interpretation and should be available at prediction time.

## Feature selection versus feature creation

- **Feature creation:** build new variables from existing data.
- **Feature selection:** keep useful variables and remove irrelevant or harmful ones.
- **Feature transformation:** change representation, such as logarithms, scaling, or encoding.

These ideas are related but not identical.

## Main lesson

> Feature engineering is where domain understanding is translated into a form that a model can learn from.
