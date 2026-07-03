# Categorical Encoding

## Why categories must be encoded

Most machine-learning algorithms cannot use text values such as `Berlin`, `Frankfurt`, or `Munich` directly. The categories must be converted into numbers.

The conversion should not create a relationship that does not exist in reality.

## Label encoding

Label encoding assigns one number to each category.

| City | Encoded value |
|---|---:|
| Berlin | 0 |
| Frankfurt | 1 |
| Munich | 2 |

This is compact, but it may create a false order. A model may treat Munich as greater than Frankfurt even though cities have no natural ranking.

Ordinal encoding is more suitable when a true order exists:

| Level | Encoded value |
|---|---:|
| Low | 0 |
| Medium | 1 |
| High | 2 |

## One-hot encoding

One-hot encoding creates one yes-or-no column for each category.

Original data:

| Customer | City |
|---|---|
| A | Berlin |
| B | Munich |
| C | Frankfurt |

Encoded data:

| Customer | city_Berlin | city_Frankfurt | city_Munich |
|---|---:|---:|---:|
| A | 1 | 0 | 0 |
| B | 0 | 0 | 1 |
| C | 0 | 1 | 0 |

Inside `city_Berlin`, a value of 1 means Berlin is present and 0 means it is absent. It does not mean Berlin is better, larger, or more important.

The computer does not understand the word Berlin. It only sees numerical columns. The column names give meaning to humans.

## High cardinality

A feature has high cardinality when it contains many unique values.

| Feature | Possible number of categories |
|---|---:|
| Weekday | 7 |
| Country | Hundreds |
| Postal code | Thousands |
| Product ID | Tens of thousands |

If a product feature has 20,000 unique values, one-hot encoding may create 20,000 mostly empty columns. This can increase memory usage and make training more difficult.

Possible approaches include:

- grouping rare categories into `Other`,
- frequency encoding,
- hashing,
- carefully designed target encoding,
- or learned embeddings.

## Example: grouping rare values

| Original category | Records | New category |
|---|---:|---|
| Electronics | 4,000 | Electronics |
| Clothing | 3,500 | Clothing |
| Books | 1,200 | Books |
| Musical instruments | 25 | Other |
| Collectibles | 12 | Other |

This reduces very small categories and keeps the representation manageable.

## Unseen categories

The training data may contain Berlin, Frankfurt, and Munich, while new data contains Cologne. The encoder should be prepared for unknown categories instead of failing.

## Choosing an encoding

| Situation | Possible choice |
|---|---|
| Unordered categories, small number | One-hot encoding |
| Categories with a true order | Ordinal encoding |
| Very many categories | Grouping or another compact representation |
| New categories expected | Encoder that handles unknown values |

## Main lesson

> Encoding should convert categories into a useful numerical representation without inventing a false order or relationship between them.
