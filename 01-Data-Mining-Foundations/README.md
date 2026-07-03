# Data Mining Foundations

## What data mining means to me

Data mining is the process of finding useful patterns, relationships, and knowledge inside data. The important point is that **having data is not the same as understanding it**.

A company may collect millions of records every day, but those records are only valuable when they help answer a real question. Data mining connects raw data with a practical decision.

For example, a factory may collect the following sensor readings:

| Time | Temperature | Vibration | Pressure | Machine status |
|---|---:|---:|---:|---|
| 10:00 | 62 | 2.1 | 118 | Normal |
| 10:05 | 70 | 3.0 | 121 | Normal |
| 10:10 | 81 | 5.8 | 127 | Failed |

The values alone are data. After studying many such records, we may discover that high temperature and high vibration often appear before a failure. That discovered relationship is useful knowledge.

## Data, information, and knowledge

I separate these three ideas as follows:

| Level | Meaning | Example |
|---|---|---|
| Data | Raw values without enough context | `81`, `5.8`, `127` |
| Information | Values connected to a meaning | Motor temperature is 81 °C |
| Knowledge | A useful pattern or conclusion | High temperature together with high vibration often appears before failure |

This distinction helped me understand why a dataset alone is not the final result. The purpose of data mining is to move from raw values toward something that supports a decision.

## A real-life example: an online shop

Suppose an online shop stores the following information:

| Customer | Visits this month | Products viewed | Items added to cart | Purchased |
|---|---:|---:|---:|---|
| A | 2 | 4 | 0 | No |
| B | 8 | 17 | 3 | Yes |
| C | 6 | 13 | 2 | Yes |
| D | 1 | 2 | 0 | No |

A useful pattern may be that customers who visit several times and add products to the cart are more likely to purchase.

The company could use this knowledge to:

- send a reminder to customers who left products in the cart,
- recommend similar products,
- estimate which visitors are likely to buy,
- or understand why some customers leave without purchasing.

The algorithm does not understand what a customer, cart, or purchase means. It only finds numerical patterns. Humans still decide whether the pattern makes business sense and how it should be used.

## Data mining is more than running a model

At first, I thought data mining mainly meant choosing an algorithm and training it. I now understand it as a complete process:

1. **Understand the problem** — What do we actually want to know or predict?
2. **Understand the data** — What does every column mean, and how was it collected?
3. **Prepare the data** — Handle missing values, text categories, different scales, and incorrect records.
4. **Choose a suitable method** — Classification, regression, clustering, association analysis, or another approach.
5. **Evaluate the result** — Does the method work on unseen data?
6. **Interpret the result** — Is it useful and reasonable in the real world?

Skipping the first steps can lead to a technically correct model that solves the wrong problem.

## Common types of data-mining tasks

| Task | Question it answers | Example |
|---|---|---|
| Classification | Which category does this record belong to? | Will a machine fail: yes or no? |
| Regression | What numerical value should be predicted? | What will the energy consumption be tomorrow? |
| Clustering | Which records naturally belong together? | Which customers have similar behaviour? |
| Association analysis | Which events or items often occur together? | Which products are commonly bought together? |
| Anomaly detection | Which records look unusual? | Is this bank transaction suspicious? |

These tasks are different, but all of them try to discover useful structure inside data.

## Why domain knowledge matters

The computer does not understand the meaning of a feature. A column called `temperature` could be renamed `x1`, and the calculation would remain the same.

Domain knowledge gives the numbers meaning. For example:

| Value | Without domain knowledge | With domain knowledge |
|---|---|---|
| Temperature = 80 | Just a large number | Possibly dangerous for this motor |
| Pressure = -15 | A numerical value | Physically impossible for this sensor |
| Customer age = 250 | An outlier | Clearly incorrect data |

This is why data mining is not only a programming task. It combines mathematics, computing, and understanding of the real problem.

## Main lesson

> Data mining is not about forcing an algorithm onto a dataset. It is about turning raw data into reliable and useful knowledge for a clearly defined purpose.
