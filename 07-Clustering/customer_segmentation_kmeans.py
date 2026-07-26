"""Beginner-friendly K-Means customer-segmentation example.

The script is self-contained: it creates customer-style numerical data,
standardises it, uses the Elbow Method, trains K-Means, evaluates the
result with a Silhouette Score, and profiles the discovered groups.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# 1. Create customer-style data
# -----------------------------------------------------------------------------
# The groups are generated only to create a realistic-looking example.
# K-Means NEVER receives the hidden group labels. It sees only income and spend.

rng = np.random.default_rng(42)

low_income_high_spend = pd.DataFrame(
    {
        "annual_income": rng.normal(30_000, 4_000, 70),
        "spending_score": rng.normal(75, 8, 70),
    }
)

middle_income_medium_spend = pd.DataFrame(
    {
        "annual_income": rng.normal(60_000, 6_000, 90),
        "spending_score": rng.normal(50, 7, 90),
    }
)

high_income_low_spend = pd.DataFrame(
    {
        "annual_income": rng.normal(95_000, 7_000, 70),
        "spending_score": rng.normal(25, 8, 70),
    }
)

customers = pd.concat(
    [
        low_income_high_spend,
        middle_income_medium_spend,
        high_income_low_spend,
    ],
    ignore_index=True,
)

# K-Means only gets numerical features.
X = customers[["annual_income", "spending_score"]]


# -----------------------------------------------------------------------------
# 2. Standardise features
# -----------------------------------------------------------------------------
# Income uses values in the tens of thousands while spending score is around
# 0-100. Scaling prevents income from dominating Euclidean distance merely
# because its raw numbers are larger.

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# -----------------------------------------------------------------------------
# 3. Elbow Method
# -----------------------------------------------------------------------------
# Inertia = WCSS = sum of squared distances from points to their own centroid.
# We try several K values and look for a point where improvement starts to slow.

k_values = range(1, 9)
inertias = []

for k in k_values:
    model = KMeans(
        n_clusters=k,
        n_init=10,        # try several initial centroid sets
        random_state=42,
    )
    model.fit(X_scaled)
    inertias.append(model.inertia_)

plt.figure()
plt.plot(list(k_values), inertias, marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("WCSS / Inertia")
plt.title("Elbow Method")
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# 4. Train the final K-Means model
# -----------------------------------------------------------------------------
# For this generated example, K=3 gives three useful customer segments.

kmeans = KMeans(
    n_clusters=3,
    n_init=10,
    random_state=42,
)

cluster_labels = kmeans.fit_predict(X_scaled)

# Attach the discovered group number to each customer.
customers["cluster"] = cluster_labels


# -----------------------------------------------------------------------------
# 5. Evaluate cluster compactness + separation
# -----------------------------------------------------------------------------
# Silhouette Score:
# close to +1 -> compact and well separated
# around 0    -> overlapping / boundary points
# negative    -> possibly poor assignment

score = silhouette_score(X_scaled, cluster_labels)
print(f"Silhouette Score: {score:.3f}\n")


# -----------------------------------------------------------------------------
# 6. Profile the discovered clusters
# -----------------------------------------------------------------------------
# Cluster numbers such as 0, 1, 2 have no human meaning by themselves.
# We inspect average behaviour so we can understand what each cluster represents.

profile = customers.groupby("cluster").agg(
    customers=("cluster", "size"),
    average_income=("annual_income", "mean"),
    average_spending_score=("spending_score", "mean"),
)

print("Cluster profile:")
print(profile.round(2))
print()


# -----------------------------------------------------------------------------
# 7. Visualise the customer segments
# -----------------------------------------------------------------------------

plt.figure()
plt.scatter(
    customers["annual_income"],
    customers["spending_score"],
    c=customers["cluster"],
)
plt.xlabel("Annual income")
plt.ylabel("Spending score")
plt.title("K-Means Customer Segmentation")
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# Human memory map
# -----------------------------------------------------------------------------
# K                    = number of clusters
# centroid             = mean location of a cluster
# fit_predict()        = learn clusters and return cluster numbers
# inertia_             = WCSS; lower means more compact clusters
# Elbow Method         = compare inertia for several K values
# silhouette_score()   = compactness + separation quality
# StandardScaler       = stop large-number features dominating distance
# n_init               = try several starting centroid configurations
# random_state         = make the example reproducible
