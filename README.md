# Machine Learning Roadmap

A beginner-friendly learning repository that explains machine learning from **normal human intuition to practical Python code**.

The goal is to make the material useful for two kinds of readers:

1. someone with little or no machine-learning background who wants to understand what the ideas actually mean;
2. a technical reviewer or employer who wants to see the concepts, workflows, code, and interpretation I have learned and practised.

This repository therefore tries to answer four questions for every topic:

```text
What problem are we solving?
        ↓
Why does this method exist?
        ↓
How does it work in normal language?
        ↓
How do we implement and interpret it in code?
```

The emphasis is not only on **how** an algorithm is used, but also **why it exists, when it is useful, what can go wrong, and how its output should be interpreted**.

## Current learning areas

### 1. [Data Mining Foundations](./01-Data-Mining-Foundations/)

- Difference between data, information, and knowledge
- Common data-mining tasks
- Importance of domain knowledge

### 2. [Data Preprocessing](./02-Data-Preprocessing/)

- Problem and domain understanding
- Sampling and generalization
- Train, validation, and test splitting
- Data leakage and overfitting
- Feature engineering
- Missing-value handling
- Feature scaling
- Categorical encoding
- Class imbalance
- Curse of dimensionality
- Principal Component Analysis
- Complete preprocessing workflow

### 3. [Decision Tree Classification and Model Evaluation](./03-Decision-Tree-Classification-and-Model-Evaluation/)

- Complete Titanic classification workflow
- Decision Tree Classifier
- Class predictions and prediction probabilities
- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- ROC Curve and AUC
- Decision-threshold comparison

Files:

- [`titanic_decision_tree_beginner.py`](./03-Decision-Tree-Classification-and-Model-Evaluation/titanic_decision_tree_beginner.py) — simple, top-to-bottom beginner version with direct explanations and manual metric calculations.
- [`titanic_decision_tree_pipeline.py`](./03-Decision-Tree-Classification-and-Model-Evaluation/titanic_decision_tree_pipeline.py) — more structured and advanced version using reusable functions.

### 4. [Explainability and Causality](./04-Explainability-and-Causality/)

- Why model explainability matters
- Global versus local explanations
- Inherent versus post-hoc explainability
- Random Forest MDI / Gini feature importance
- Permutation Feature Importance
- SHAP / Shapley values
- Global SHAP importance
- SHAP beeswarm interpretation
- Predictive importance versus causality
- Causal inference, counterfactuals, interventions, and confounders
- Titanic explainability workflow with recorded exercise outputs

Files:

- [`README.md`](./04-Explainability-and-Causality/README.md) — concepts, code snippets, interpretation, and recorded exercise outputs.
- [`titanic_explainability.py`](./04-Explainability-and-Causality/titanic_explainability.py) — executable Python version.

### 5. [Large Language Models and RAG](./05-LLMs-and-RAG/)

- Generative AI intuition
- Text as sequence data and why context matters
- Tokenization and self-supervised next-token prediction
- LLM output probabilities and vocabulary size
- Autoregressive text generation
- Greedy decoding and temperature-based sampling
- LLM versus complete chatbot applications
- LLM limitations and hallucination
- Retrieval-Augmented Generation (RAG)
- RAG versus fine-tuning
- Document chunking and metadata
- Sentence embeddings
- Vector databases and semantic search
- Cosine similarity and other distance metrics
- Pinecone retrieval
- Prompt augmentation
- Local Llama generation with Ollama
- Complete RAG workflow with recorded course-exercise outputs

Files:

- [`README.md`](./05-LLMs-and-RAG/README.md) — LLM theory, RAG concepts, practical workflow, outputs, and exam cheat sheet.
- [`rag_pipeline.py`](./05-LLMs-and-RAG/rag_pipeline.py) — executable open-source RAG pipeline based on the course exercise.

### 6. [Linear Regression](./06-Linear-Regression/)

- Continuous-value prediction explained with everyday examples
- Features, target, weights, and bias
- Model fitting and Linear Regression assumptions
- Loss versus cost / objective functions
- Mean Squared Error
- Gradient and Gradient Descent using a human hill/valley analogy
- Learning rate and convergence
- Multiple parameters, partial derivatives, and chain-rule intuition
- Convex versus non-convex optimization
- Polynomial Regression
- Ridge, Lasso, and Elastic Net regularization
- R², MSE, coefficients, scaling, and GridSearchCV
- Scikit-learn and from-scratch Gradient Descent examples

Files:

- [`README.md`](./06-Linear-Regression/README.md) — concept-first Linear Regression and Gradient Descent guide.
- [`linear_regression.py`](./06-Linear-Regression/linear_regression.py) — self-contained building-energy regression example with scaling, evaluation, Lasso, and GridSearchCV.

### 7. [Clustering](./07-Clustering/)

- Clustering versus classification
- Unsupervised learning intuition
- K-Means step by step
- Centroids and Euclidean distance
- Random initialisation and local optima
- WCSS / inertia
- Elbow Method
- Gap Statistic
- K-Means limitations
- Silhouette Score
- Cluster profiling and interpretation

Files:

- [`README.md`](./07-Clustering/README.md) — clustering explained from everyday intuition to K-Means terminology and limitations.
- [`customer_segmentation_kmeans.py`](./07-Clustering/customer_segmentation_kmeans.py) — self-contained customer-segmentation example with scaling, Elbow Method, K-Means, Silhouette Score, and cluster profiling.

### 8. [Logistic Regression](./08-Logistic-Regression/)

- Why Logistic Regression is used for classification
- Linear score `z = w^T x + b`
- Sigmoid function and probability interpretation
- Decision thresholds
- Binary Cross-Entropy / log loss
- Gradient Descent and chain-rule intuition
- Connection to backpropagation
- One-vs-Rest multiclass classification
- Softmax and multiclass probabilities
- Sigmoid versus Softmax
- Logistic Regression limitations and linear decision boundaries
- Bridge from Logistic Regression to Neural Networks
- Scikit-learn and from-scratch implementations

Files:

- [`README.md`](./08-Logistic-Regression/README.md) — concept-first Logistic Regression guide.
- [`logistic_regression.py`](./08-Logistic-Regression/logistic_regression.py) — executable scikit-learn and from-scratch examples.

### 9. [Neural Networks](./09-Neural-Networks/)

- Artificial neuron explained as weighted evidence
- Weights, bias, and activation functions
- Hidden layers and hierarchical feature learning
- ReLU, Sigmoid, and Softmax
- Forward propagation
- Loss functions
- Backpropagation and the chain rule
- Gradient Descent and learning rate
- Batch, stochastic, and mini-batch training
- Adam optimizer
- Epochs and batch size
- Input normalisation
- Overfitting, dropout, and early stopping
- Vanishing and exploding gradients
- Saddle points and optimisation difficulty
- Complete neural-network training workflow

Files:

- [`README.md`](./09-Neural-Networks/README.md) — neural networks explained from normal-human intuition to training mechanics.
- [`neural_network_cifar10.py`](./09-Neural-Networks/neural_network_cifar10.py) — CIFAR-10 multiclass image-classification example with training curves, evaluation, and learning-rate comparison.

---

## Repository philosophy

The roadmap is intentionally written as a learning record rather than a collection of unexplained code snippets.

A reader should be able to enter a topic folder and understand:

```text
problem
→ intuition
→ terminology
→ mathematics
→ code
→ output
→ interpretation
→ limitations
```

More practical examples, visualisations, and machine-learning topics will be added as the roadmap develops.
