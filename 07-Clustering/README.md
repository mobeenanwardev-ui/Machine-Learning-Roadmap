# Clustering

This section explains clustering for a reader who may have **never studied machine learning before**.

The goal is not to start with formulas. The goal is to start with a normal human question:

> If nobody tells us the groups in advance, can a computer discover which things naturally look similar?

That is the idea behind clustering.

---

# 1. A normal-life example

Imagine a shop has 10,000 customers.

For every customer, the shop knows things such as:

- age,
- annual income,
- number of purchases,
- average order value,
- spending score.

But nobody has labelled the customers as:

- loyal customers,
- occasional customers,
- high-value customers,
- low-spending customers.

Clustering tries to discover these groups automatically from the data.

```text
Customer data
     ↓
No predefined labels
     ↓
Find customers that behave similarly
     ↓
Create groups / clusters
     ↓
Study what each group means
```

The important idea is:

> Clustering discovers structure. It does not begin with known class labels.

---

# 2. Clustering vs classification

These two are easy to confuse.

## Classification

In classification, the classes are already known.

Example:

```text
Email
  ↓
Model
  ↓
Spam / Not Spam
```

The training data contains the correct answers.

This is **supervised learning**.

## Clustering

In clustering, the groups are not given in advance.

Example:

```text
Customer behaviour
       ↓
Clustering algorithm
       ↓
Discover natural customer groups
```

This is **unsupervised learning**.

A useful memory rule:

```text
Classification = predict known groups
Clustering     = discover unknown groups
```

---

# 3. What is K-Means?

K-Means is one of the most common clustering algorithms.

The name tells us two important things:

```text
K     = number of clusters
Means = cluster centres are calculated using the mean
```

Suppose we choose:

```text
K = 3
```

We are asking K-Means to divide the data into three groups.

Important:

> Standard K-Means requires us to choose K before the algorithm runs.

---

# 4. The K-Means algorithm in normal language

K-Means repeats two main actions:

1. assign each point to the nearest centre,
2. move each centre to the mean of the points assigned to it.

The complete process is:

```text
Choose K
   ↓
Choose K starting centroids
   ↓
Measure distance from each point to the centroids
   ↓
Assign every point to the nearest centroid
   ↓
Recalculate each centroid as the mean of its cluster
   ↓
Did the assignments change?
   ↓
Yes → repeat
No  → stop
```

It may also stop after a configured maximum number of iterations.

---

# 5. What is a centroid?

A centroid is the centre of a cluster.

For example, imagine one-dimensional points:

```text
4, 5, 6
```

Their mean is:

```text
(4 + 5 + 6) / 3 = 5
```

So the centroid is `5`.

In multiple dimensions, the mean is calculated separately for every feature.

A useful memory rule:

```text
Centroid = average location of the points in a cluster
```

---

# 6. How does K-Means decide the nearest cluster?

K-Means normally relies on Euclidean distance.

For two dimensions:

```text
distance² = (x1 - c1)² + (x2 - c2)²
```

where:

```text
x = data point
c = centroid
```

The point is assigned to the centroid with the smallest distance.

Conceptually:

```text
Point
  ↓
Measure distance to every centroid
  ↓
Choose nearest centroid
  ↓
Assign cluster
```

---

# 7. Why can two K-Means runs give different answers?

The starting centroids may be chosen randomly.

That means:

```text
Run 1 → starting centres A
Run 2 → starting centres B
```

The algorithm may follow different paths and finish with different cluster assignments.

This is why K-Means is not necessarily deterministic unless the random seed and configuration are fixed.

---

# 8. Local optimum

K-Means improves the solution from its current starting position.

A useful real-life analogy is walking downhill at night.

You keep walking downward until there is nowhere lower nearby.

You may have reached a valley, but it might not be the lowest valley in the entire landscape.

```text
Local optimum  = best solution nearby
Global optimum = best solution overall
```

K-Means can get stuck in a local optimum because of unlucky initial centroids.

A practical solution is to run K-Means several times with different initialisations and keep the best result.

---

# 9. WCSS / Inertia

We need a way to measure how compact the clusters are.

That measure is commonly called:

- **WCSS**: Within-Cluster Sum of Squares,
- **inertia** in scikit-learn.

For every point:

1. measure its distance to its own centroid,
2. square the distance,
3. add all squared distances together.

Conceptually:

```text
point-to-centroid squared distances
              ↓
             sum
              ↓
        WCSS / inertia
```

Interpretation:

```text
Low WCSS  → points are close to their centroids
High WCSS → clusters are more spread out
```

K-Means tries to minimise WCSS.

---

# 10. Why not simply keep increasing K?

Because WCSS normally decreases as K increases.

In the extreme case, if every data point had its own cluster, the distance from every point to its centroid would be zero.

So we cannot simply say:

> Choose the K with the smallest WCSS.

That would usually push us toward too many clusters.

We need a trade-off between:

- compact clusters,
- a sensible number of groups.

That leads to the Elbow Method.

---

# 11. Elbow Method

The Elbow Method tries several K values.

For example:

```text
K = 1, 2, 3, 4, 5, 6, ...
```

For each K:

1. run K-Means,
2. record WCSS / inertia,
3. plot K against inertia.

Example idea:

```text
WCSS
│\
│ \
│  \
│   \____
│        \___
└────────────── K
        ↑
      elbow
```

Before the elbow, adding a cluster gives a large improvement.

After the elbow, adding more clusters gives only a small improvement.

So the elbow is a practical trade-off.

Important limitation:

> The elbow can be subjective. Different people may choose slightly different points.

---

# 12. Gap Statistic

The Gap Statistic asks a different question:

> Is the clustering in the real data stronger than what we could get from random data by chance?

For each K:

```text
Real data
  ↓
K-Means
  ↓
Real-data WCSS
```

and:

```text
Random reference data
        ↓
      K-Means
        ↓
Random-data WCSS
```

Then the results are compared.

The random reference experiment is repeated several times because one random sample can be misleading.

Interpretation:

```text
Large Gap
→ real data clusters much better than random data
→ stronger evidence of meaningful structure

Small Gap
→ real-data clustering is not much better than random expectation
```

Important correction:

> A small Gap does not directly mean two neighbouring clusters are physically close. It means the real clustering is not much better than what random data could produce.

---

# 13. Elbow vs Gap Statistic

```text
ELBOW METHOD
-------------
Uses real-data inertia
Looks for diminishing improvement
Easy to understand
Can be subjective

GAP STATISTIC
-------------
Compares real clustering with random reference data
Asks whether structure is stronger than chance
More systematic
Requires extra computation
```

---

# 14. Main limitations of K-Means

K-Means is useful, but it is not suitable for every dataset.

## 14.1 Outliers

A centroid is a mean.

Means can be pulled strongly by extreme values.

```text
Outlier
   ↓
Moves the mean
   ↓
Moves the centroid
   ↓
Can distort the cluster
```

## 14.2 Feature scaling

Suppose we use:

```text
Age:    18 to 70
Income: 20,000 to 100,000
```

Distance calculations can become dominated by income simply because the numbers are much larger.

A common solution is feature standardisation.

```text
StandardScaler
→ put features on comparable scales
```

## 14.3 Irregular cluster shapes

K-Means works best with compact, roughly round / blob-like clusters.

It can struggle with:

- rings,
- moon shapes,
- long curved groups,
- strongly irregular structures.

## 14.4 Different cluster sizes or densities

K-Means may struggle when one cluster is very large and another is very small, or when their densities are very different.

## 14.5 Categorical features

Classic K-Means relies on numerical distance.

A category such as:

```text
red, blue, green
```

does not naturally have a Euclidean distance.

Giving them arbitrary numbers such as `1, 2, 3` can create fake distance relationships.

## 14.6 Random initialisation

Poor starting centroids can lead to a poor local solution.

## 14.7 Large or high-dimensional datasets

Repeated distance calculations can become expensive, and high-dimensional clusters can be difficult to interpret.

Variants such as **Mini-Batch K-Means** are useful when ordinary K-Means becomes expensive.

---

# 15. How can clustering be evaluated?

There may be no correct labels, so ordinary classification accuracy often does not apply.

One useful metric is the **Silhouette Coefficient**.

It asks:

```text
Is a point close to its own cluster?
AND
Is it far from neighbouring clusters?
```

Typical interpretation:

```text
close to +1 → compact and well separated
around 0    → near a cluster boundary
negative    → possibly assigned to a poor cluster
```

Other clustering metrics exist, but every metric makes assumptions.

A good score is not the whole story.

---

# 16. Interpretation matters more than cluster numbers

K-Means may output:

```text
Cluster 0
Cluster 1
Cluster 2
```

These numbers have no business meaning by themselves.

We should profile the clusters.

For customer segmentation, we might calculate:

- average income,
- average spending,
- median number of purchases,
- most common behaviour,
- features that differ most between groups.

Then we might give meaningful names:

```text
Cluster 0 → high-value loyal customers
Cluster 1 → occasional low spenders
Cluster 2 → new high-potential customers
```

This is often where clustering becomes genuinely useful.

> Clustering is exploratory. The final interpretation of the discovered groups matters.

---

# 17. Practical Python workflow

The accompanying script [`customer_segmentation_kmeans.py`](./customer_segmentation_kmeans.py) demonstrates a complete beginner-friendly workflow:

```text
Create customer-style data
        ↓
Scale features
        ↓
Try several K values
        ↓
Plot Elbow curve
        ↓
Train K-Means
        ↓
Calculate Silhouette Score
        ↓
Attach cluster labels
        ↓
Profile each cluster
        ↓
Visualise the segments
```

The code uses generated data so it runs without requiring an external dataset download.

---

# 18. Quick memory sheet

```text
Clustering
→ discover unknown groups
→ unsupervised learning

K-Means
→ choose K
→ initialise centroids
→ assign points to nearest centroid
→ centroid becomes cluster mean
→ repeat until stable

WCSS / inertia
→ sum of squared distances to own centroid
→ lower = more compact

Elbow
→ try several K values
→ find point where extra clusters stop helping much

Gap Statistic
→ compare real clustering against random reference data
→ large gap = stronger-than-random structure

Silhouette
→ own-cluster compactness + separation from other clusters

Main K-Means problems
→ outliers
→ scaling
→ irregular shapes
→ unequal cluster sizes
→ categorical data
→ local optima
```

---

# Final human explanation

Imagine entering a room full of people you have never met.

Nobody gives you labels such as "students", "teachers", "engineers", or "tourists".

You start observing similarities and naturally notice groups.

That is the human intuition behind clustering.

K-Means is one mathematical way of doing it:

> place centres, assign nearby points, move the centres to the group averages, and repeat until the groups stabilise.
