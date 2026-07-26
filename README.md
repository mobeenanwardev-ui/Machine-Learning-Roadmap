# Machine Learning Roadmap

A structured learning repository covering machine learning concepts from theory to practical implementation.

The goal of this repository is to document not only how techniques are used, but why they exist, what problems they solve, when they fail, and how they are applied in real-world projects.

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

This section contains two executable Python versions:

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

- Linear Regression intuition and model equation
- Features, target, weights, and bias
- Model fitting and Linear Regression assumptions
- Loss versus cost / objective functions
- Mean Squared Error and the quadratic cost surface
- Gradient and Gradient Descent using a human hill/valley analogy
- Learning rate and convergence
- Multiple parameters, partial derivatives, and chain-rule intuition
- Convex versus non-convex optimization
- Polynomial Regression
- Ridge, Lasso, and Elastic Net regularization
- Scikit-learn and from-scratch Gradient Descent examples
- Exam-style questions and a compact revision cheat sheet

Files:

- [`README.md`](./06-Linear-Regression/README.md) — concept-first Linear Regression and Gradient Descent study guide designed for fast exam revision.

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
- Exam-style questions and a compact revision cheat sheet

Files:

- [`README.md`](./08-Logistic-Regression/README.md) — concept-first Logistic Regression revision guide.
- [`logistic_regression.py`](./08-Logistic-Regression/logistic_regression.py) — executable scikit-learn and from-scratch examples.

More practical notebooks, Python scripts, visualizations, and machine-learning topics will be added as the roadmap develops.
