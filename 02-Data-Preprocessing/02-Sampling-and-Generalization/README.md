# Sampling and Generalization

## What I understood

Usually, we cannot collect data from every person, machine, transaction, or event in the real world. Instead, we work with a sample and hope that what we learn from it also works on the larger population.

The sample size matters, but size alone is not enough. A very large biased sample can still produce a bad model.

For example, if I want to build a model for all university students but collect data only from computer-science students, the sample may be large but not representative. The model may learn patterns that do not apply to students from other subjects.

## Population, sample, and representativeness

- **Population:** the full group I want to make conclusions about.
- **Sample:** the smaller group actually present in my dataset.
- **Representative sample:** a sample that contains the important variation of the population.

A representative sample should cover relevant differences such as age groups, locations, machine types, seasons, operating conditions, or customer categories.

## Generalization

Generalization means that a model performs well on new examples that were not used during training.

A model that performs perfectly on training data but badly on new data has not learned the general pattern. It has learned details specific to the training sample.

Suppose a failure-prediction model is trained only with winter data. It may fail during summer because temperature and operating conditions are different. The model may look accurate during development, but the sample did not represent the real environment.

## Does more data always help?

More relevant and representative data usually helps. However, simply adding more records does not automatically solve:

- biased sampling,
- incorrect labels,
- duplicated observations,
- outdated data,
- or missing important situations.

Quality and coverage are as important as quantity.

## Main lesson

> A model can only learn from the situations included in its data. If the sample does not represent reality, the model will not reliably generalize to reality.
