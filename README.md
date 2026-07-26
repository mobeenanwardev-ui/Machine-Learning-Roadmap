# Machine Learning Roadmap

A beginner-friendly learning repository that explains machine learning from **normal human intuition to practical Python code**.

The repository is arranged as a learning path rather than by the order in which topics happened to be studied. A beginner can start at `01` and move forward step by step, while a technical reviewer or employer can quickly see the progression from foundations to modern AI systems.

The goal is to make the material useful for two kinds of readers:

1. someone with little or no machine-learning background who wants to understand what the ideas actually mean;
2. a technical reviewer or employer who wants to see the concepts, workflows, code, and interpretation I have learned and practised.

Every topic tries to answer the same questions:

```text
What problem are we solving?
        ↓
Why does this method exist?
        ↓
How does it work in normal language?
        ↓
What is the technical terminology / mathematics?
        ↓
How do we implement it in Python?
        ↓
How do we interpret the result?
        ↓
What are the limitations?
```

---

# Recommended Learning Order

```text
01. Data Mining Foundations
        ↓
02. Data Preprocessing
        ↓
03. Linear Regression
        ↓
04. Logistic Regression
        ↓
05. Decision Trees & Model Evaluation
        ↓
06. Clustering
        ↓
07. Explainability & Causality
        ↓
08. Neural Networks
        ↓
09. Large Language Models & RAG
```

This gives the repository a natural progression:

```text
FOUNDATIONS
    ↓
PREPARE DATA
    ↓
SUPERVISED MACHINE LEARNING
    ↓
UNSUPERVISED MACHINE LEARNING
    ↓
UNDERSTAND / EXPLAIN MODELS
    ↓
DEEP LEARNING
    ↓
GENERATIVE AI
```

---

# Phase 1 — Foundations and Data Preparation

## 01. [Data Mining Foundations](./01-Data-Mining-Foundations/)

Start here before thinking about algorithms.

Topics include:

- difference between data, information, and knowledge;
- common data-mining tasks;
- why domain knowledge matters;
- the purpose of discovering useful patterns in data.

## 02. [Data Preprocessing](./02-Data-Preprocessing/)

Real machine-learning work begins with understanding and preparing the data.

Topics include:

- problem and domain understanding;
- sampling and generalisation;
- train, validation, and test splitting;
- data leakage and overfitting;
- feature engineering;
- missing-value handling;
- feature scaling;
- categorical encoding;
- class imbalance;
- curse of dimensionality;
- Principal Component Analysis (PCA);
- complete preprocessing workflow.

---

# Phase 2 — Supervised Machine Learning

Supervised learning means the training data contains a target / correct answer that the model tries to learn.

## 03. [Linear Regression](./03-Linear-Regression/)

The first predictive model in the roadmap: predict a **continuous numerical value**.

Examples include house price, temperature, salary, or heating demand.

Topics include:

- Linear Regression intuition and equation;
- features, target, weights, and bias;
- model fitting;
- assumptions of Linear Regression;
- loss versus cost / objective functions;
- Mean Squared Error;
- Gradient Descent and learning rate;
- partial derivatives and chain-rule intuition;
- convex versus non-convex optimisation;
- Polynomial Regression;
- Ridge, Lasso, and Elastic Net;
- R² and MSE;
- coefficients and interpretation;
- StandardScaler;
- GridSearchCV.

Files:

- [`README.md`](./03-Linear-Regression/README.md) — concept-first Linear Regression and Gradient Descent guide.
- [`linear_regression.py`](./03-Linear-Regression/linear_regression.py) — self-contained building-energy regression example.

## 04. [Logistic Regression](./04-Logistic-Regression/)

Move from predicting a number to predicting a **class**.

Topics include:

- why Logistic Regression is used for classification;
- linear score `z = w^T x + b`;
- Sigmoid function;
- probability interpretation;
- decision thresholds;
- Binary Cross-Entropy / log loss;
- Gradient Descent and chain rule;
- One-vs-Rest multiclass classification;
- Softmax;
- Sigmoid versus Softmax;
- limitations and linear decision boundaries;
- bridge from Logistic Regression to Neural Networks.

Files:

- [`README.md`](./04-Logistic-Regression/README.md) — concept-first Logistic Regression guide.
- [`logistic_regression.py`](./04-Logistic-Regression/logistic_regression.py) — scikit-learn and from-scratch examples.

## 05. [Decision Tree Classification and Model Evaluation](./05-Decision-Tree-Classification-and-Model-Evaluation/)

After understanding basic classification, this section shows another major classification model and how to evaluate classifiers properly.

Topics include:

- complete Titanic classification workflow;
- Decision Tree Classifier;
- class predictions and probabilities;
- Confusion Matrix;
- Accuracy;
- Precision;
- Recall;
- F1 Score;
- Classification Report;
- ROC Curve and AUC;
- decision-threshold comparison.

Files:

- [`titanic_decision_tree_beginner.py`](./05-Decision-Tree-Classification-and-Model-Evaluation/titanic_decision_tree_beginner.py) — simple top-to-bottom beginner implementation.
- [`titanic_decision_tree_pipeline.py`](./05-Decision-Tree-Classification-and-Model-Evaluation/titanic_decision_tree_pipeline.py) — structured reusable pipeline.

---

# Phase 3 — Unsupervised Machine Learning

## 06. [Clustering](./06-Clustering/)

Unlike supervised learning, clustering starts without predefined target labels and tries to discover natural groups in the data.

Topics include:

- clustering versus classification;
- unsupervised learning intuition;
- K-Means step by step;
- centroids and Euclidean distance;
- random initialisation and local optima;
- WCSS / inertia;
- Elbow Method;
- Gap Statistic;
- K-Means limitations;
- Silhouette Score;
- cluster profiling and interpretation.

Files:

- [`README.md`](./06-Clustering/README.md) — clustering from everyday intuition to terminology, evaluation, and limitations.
- [`customer_segmentation_kmeans.py`](./06-Clustering/customer_segmentation_kmeans.py) — customer-segmentation example with scaling, Elbow Method, K-Means, Silhouette Score, and profiling.

---

# Phase 4 — Understanding What Models Learned

## 07. [Explainability and Causality](./07-Explainability-and-Causality/)

A model can make a prediction, but we also need to understand **why** it made that prediction — and why prediction is not automatically causation.

Topics include:

- why model explainability matters;
- global versus local explanations;
- inherent versus post-hoc explainability;
- Random Forest MDI / Gini feature importance;
- Permutation Feature Importance;
- SHAP / Shapley values;
- global SHAP importance;
- SHAP beeswarm interpretation;
- predictive importance versus causality;
- causal inference;
- counterfactuals;
- interventions;
- confounders.

Files:

- [`README.md`](./07-Explainability-and-Causality/README.md) — concepts, code, interpretation, and exercise outputs.
- [`titanic_explainability.py`](./07-Explainability-and-Causality/titanic_explainability.py) — executable Titanic explainability workflow.

---

# Phase 5 — Deep Learning

## 08. [Neural Networks](./08-Neural-Networks/)

This section builds from the weighted-sum idea used in regression to multi-layer networks that can learn complex non-linear representations.

Topics include:

- artificial neuron explained as weighted evidence;
- weights, bias, and activation functions;
- hidden layers and hierarchical feature learning;
- ReLU, Sigmoid, and Softmax;
- forward propagation;
- loss functions;
- backpropagation and the chain rule;
- Gradient Descent and learning rate;
- batch, stochastic, and mini-batch training;
- Adam optimizer;
- epochs and batch size;
- input normalisation;
- overfitting, dropout, and early stopping;
- vanishing and exploding gradients;
- saddle points and optimisation difficulty;
- complete neural-network training workflow.

Files:

- [`README.md`](./08-Neural-Networks/README.md) — neural networks from normal-human intuition to training mechanics.
- [`neural_network_cifar10.py`](./08-Neural-Networks/neural_network_cifar10.py) — CIFAR-10 image-classification example with training curves and learning-rate experiments.

---

# Phase 6 — Generative AI

## 09. [Large Language Models and RAG](./09-LLMs-and-RAG/)

This section comes after the neural-network foundations because modern LLMs are deep-learning systems.

Topics include:

- Generative AI intuition;
- text as sequence data and why context matters;
- tokenisation;
- self-supervised next-token prediction;
- output probabilities and vocabulary size;
- autoregressive generation;
- greedy decoding and temperature sampling;
- LLM limitations and hallucination;
- Retrieval-Augmented Generation (RAG);
- RAG versus fine-tuning;
- chunking and metadata;
- embeddings;
- vector databases and semantic search;
- cosine similarity;
- Pinecone retrieval;
- prompt augmentation;
- local Llama generation with Ollama.

Files:

- [`README.md`](./09-LLMs-and-RAG/README.md) — LLM theory, RAG concepts, practical workflow, and revision material.
- [`rag_pipeline.py`](./09-LLMs-and-RAG/rag_pipeline.py) — executable open-source RAG pipeline based on the course exercise.

---

# Repository Philosophy

This repository is intentionally a **learning record**, not a collection of unexplained code snippets.

A reader should be able to enter a topic folder and follow:

```text
problem
→ normal-life intuition
→ terminology
→ mathematics
→ code
→ output
→ interpretation
→ limitations
```

The numbering now represents the recommended learning order, so the repository can be read like a small machine-learning course from top to bottom.
