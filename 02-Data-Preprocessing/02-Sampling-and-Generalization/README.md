# Sampling and Generalization

## Population and sample

In most real problems, I cannot collect data from every person, machine, transaction, or event. The complete group is called the **population**, while the smaller part used for analysis is called the **sample**.

| Term | Meaning | Example |
|---|---|---|
| Population | Every case I want to make conclusions about | All university students in Germany |
| Sample | The cases available in my dataset | 2,000 students who completed a survey |
| Representative sample | A sample that reflects important differences in the population | Students from different cities, subjects, ages, and study levels |

The goal is to learn something from the sample that also works for the population.

## Why a large sample can still be bad

Sample size matters, but size alone is not enough.

Suppose I want to understand all university students, but I collect data only from computer-science students:

| Subject | Students in population | Students in sample |
|---|---:|---:|
| Computer Science | 20% | 100% |
| Business | 25% | 0% |
| Engineering | 25% | 0% |
| Social Sciences | 30% | 0% |

Even if the sample contains 50,000 students, it is still biased. It does not represent the full population.

This taught me that **10,000 biased observations can be worse than 1,000 representative observations**.

## Generalization

Generalization means that a model works well on new cases that were not used during training.

Consider a machine-failure model trained only during winter:

| Training conditions | Real deployment conditions |
|---|---|
| Low room temperature | Summer heat |
| Normal workload | Peak production workload |
| One machine model | Several machine models |
| Mostly new machines | New and old machines |

The model may perform well during development but fail after deployment because it never learned the missing situations.

A model can only learn from the variation present in its data.

## More data does not automatically fix everything

Adding more records is helpful when the new data improves coverage and quality. It does not automatically solve:

- incorrect labels,
- duplicated observations,
- outdated records,
- missing groups,
- measurement errors,
- or sampling bias.

For example:

| Dataset change | Likely benefit |
|---|---|
| Add 100,000 duplicated records | Very little |
| Add summer operating data | Better seasonal coverage |
| Add examples from older machines | Better machine-age coverage |
| Add records with incorrect failure labels | May make learning worse |

## Random sampling is not always enough

Different datasets need different sampling strategies.

| Situation | Better approach |
|---|---|
| Rare failure cases | Stratified sampling to preserve class ratios |
| Time-series data | Keep older data before newer data |
| Several records from one patient | Keep each patient in only one split |
| Different machine models | Make sure all important models are represented |

The sampling method should match the structure of the real problem.

## Questions I should ask

1. Which population should this model work for?
2. Which groups or conditions are missing from the dataset?
3. Is the class distribution realistic?
4. Does the data cover different seasons, locations, machine types, or customer groups?
5. Was the sample collected in a way that favours some cases over others?

## Main lesson

> A model can only learn from the situations included in its data. If the sample does not represent reality, the model will not reliably generalize to reality.
