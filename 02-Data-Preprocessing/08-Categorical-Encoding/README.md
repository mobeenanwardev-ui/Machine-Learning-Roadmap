# Categorical Encoding

## What I understood

Most machine-learning algorithms cannot use text categories directly. Values such as `Berlin`, `Frankfurt`, and `Munich` must be converted into numbers, but the conversion should not create a meaning that is not present in the data.

## Label encoding

Label encoding gives each category a number. This is compact, but for unordered categories it can create a false order. For example, assigning Berlin = 0, Frankfurt = 1, and Munich = 2 may make the model treat Munich as greater than Frankfurt even though cities have no natural ranking.

It is more suitable when a real order exists, such as low, medium, and high.

## One-hot encoding

One-hot encoding creates a separate yes-or-no column for each category. A row belonging to Berlin gets a 1 in `city_Berlin` and 0 in the other city columns.

The computer does not understand the word Berlin. It only reads the numerical pattern. The column name gives meaning to humans.

## High cardinality

A feature has high cardinality when it contains many unique values, such as product IDs or postal codes. One-hot encoding can then create thousands of mostly empty columns.

Possible alternatives include grouping rare categories, frequency encoding, hashing, target encoding, or embeddings. These methods must still be validated carefully, especially when the target is used during encoding.

## Unseen categories

A real system may receive a category that was not present during training. The preprocessing pipeline should handle unknown values instead of failing.

## Main lesson

> Encoding should turn categories into a useful numerical form without inventing a false relationship between them.

## How I would explain it in an interview

I use one-hot encoding for unordered categories with manageable cardinality and ordinal encoding when a true order exists. For high-cardinality data, I consider other encodings and check memory use, leakage, and unseen categories.