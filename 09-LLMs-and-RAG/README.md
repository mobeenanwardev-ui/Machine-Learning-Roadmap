# Large Language Models and Retrieval-Augmented Generation

This section documents the lecture on **Generative AI, Large Language Models (LLMs), and Retrieval-Augmented Generation (RAG)** and connects the theory to the open-source RAG exercise using LangChain, Sentence Transformers, Pinecone, and Ollama.

The goal is the same as the rest of this roadmap:

> Understand **what is happening, why it is happening, and what the code is actually doing**.

## Files in this section

- [`README.md`](./README.md) - concepts, diagrams, exercise outputs, interpretation, and compact exam notes.
- [`rag_pipeline.py`](./rag_pipeline.py) - a complete Python version of the RAG workflow used in the course exercise.

---

# 1. Generative AI - the main intuition

Traditional supervised models often learn a mapping such as:

```text
Input
  ↓
Classification / Regression
  ↓
Prediction
```

A generative model has a different goal. It tries to learn patterns in the training-data distribution well enough that it can create new samples that look as if they could have come from that distribution.

```text
Training examples
      ↓
Learn patterns / distribution
      ↓
Sample from learned distribution
      ↓
Generate new data
```

For language, the generated data is text.

For images, the generated data can be pixels.

The key intuition is:

> **Generation requires learning what realistic data looks like, not only assigning an existing sample to a class.**

---

# 2. Why text is difficult

Text is sequence data, but it is more difficult to represent than a normal table.

Important reasons:

- **Unstructured** - there is no fixed numerical column structure.
- **Ambiguous** - the same word can have different meanings.
- **Context-dependent** - surrounding words can completely change meaning.

Example:

```text
"bank"
```

could refer to a financial institution or the side of a river.

So a language model must use **context**, not only isolated words.

---

# 3. Language modelling

A language model learns to predict the next token from the tokens that came before it.

Conceptually:

```text
Context: "Honey, can you pass me the"

Possible next tokens:

gun      → probability
pepper   → probability
salt     → probability
```

The surrounding context changes those probabilities.

Mathematically, the idea is:

```text
P(next token | previous context)
```

This is why language modelling can become a general-purpose foundation for many NLP tasks.

Tasks such as:

- summarization,
- sentiment analysis,
- question answering,
- text classification,
- translation,
- and text generation

can all be expressed as language-in / language-out problems with the right prompt.

---

# 4. How an LLM is trained

The lecture simplified LLM training into four main steps.

## Step 1 - collect a huge text corpus

The model is trained on very large quantities of text.

```text
Web pages
Wikipedia
Books
Code
Other text
    ↓
Large training corpus
```

## Step 2 - tokenization

An LLM does **not** directly receive normal words.

The text is split into **tokens**.

Tokens may be:

- words,
- subwords,
- characters,
- or bytes,

depending on the tokenizer.

```text
"playing"
   ↓
possible tokens
"play" + "ing"
```

The tokenizer maps those tokens to numerical IDs.

```text
Text
 ↓
Tokenizer
 ↓
Token IDs
```

The model therefore works with numbers rather than raw strings.

## Step 3 - next-token prediction

During training, the model repeatedly learns a problem like:

```text
Input:
"The cat sat on the"

Correct next token:
"mat"
```

The model predicts probabilities for possible next tokens.

Its prediction is compared with the correct token, an error is calculated, and the model parameters are updated through training/backpropagation.

This is **self-supervised learning** because the text itself supplies the target.

We do not need a human to manually label every sentence.

## Step 4 - repeat at massive scale

The process is repeated across enormous amounts of text.

Over time, the model learns statistical patterns involving:

- grammar,
- word order,
- semantics,
- relationships between concepts,
- styles,
- and many recurring structures in language.

### Training memory rule

```text
TEXT
 ↓
TOKENIZE
 ↓
PREDICT NEXT TOKEN
 ↓
COMPARE WITH TRUE TOKEN
 ↓
UPDATE WEIGHTS
 ↓
REPEAT
```

---

# 5. What does the LLM output?

The model does not directly output a sentence in one step.

Its output layer produces a score for the possible tokens in its vocabulary.

After softmax, these scores can be interpreted as probabilities.

```text
Model output
     ↓
Softmax
     ↓
Probability for each possible token
```

If the vocabulary contains `V` tokens, the output probability vector has approximately `V` positions.

Example:

```text
Vocabulary size = 50,000

Output
→ probability vector with 50,000 token probabilities
```

The lecture quiz therefore connects two ideas:

```text
Output layer
→ probabilities for tokens

Output-layer size
→ vocabulary size
```

---

# 6. How an LLM generates a full answer - autoregression

Suppose the prompt is:

```text
An apple a day keeps the
```

The model predicts probabilities for the next token.

A decoding strategy chooses one token, for example:

```text
doctor
```

That token is then added back to the context:

```text
An apple a day keeps the doctor
```

The model predicts again.

This repeats token by token.

```text
Prompt
  ↓
Predict token
  ↓
Append token to context
  ↓
Predict next token
  ↓
Append
  ↓
Repeat
```

This iterative generation is called **autoregression**.

Important:

> **An LLM generates text one token at a time.**

---

# 7. Greedy decoding and temperature

## Greedy decoding

The simplest strategy is:

> Always select the token with the highest probability.

```text
salt   0.54  ← choose
pepper 0.33
gun    0.13
```

This is deterministic for the same model and context.

The problem is that always selecting the maximum can make generation repetitive and less diverse.

## Temperature-based sampling

Temperature changes the probability distribution before sampling.

```text
Low temperature
→ sharper probabilities
→ more deterministic

High temperature
→ flatter probabilities
→ more randomness / diversity
```

Memory rule:

```text
T < 1  → safer / more deterministic
T = 1  → normal distribution
T > 1  → more random / creative
```

The important distinction is:

```text
Model
→ produces probabilities

Decoding strategy
→ decides how a token is selected from those probabilities
```

---

# 8. An LLM is not the whole chatbot

An LLM is the language-generating model.

A practical chatbot/application usually adds other components around it, such as:

```text
User interface
Prompt / instruction handling
Conversation context
Alignment / fine-tuning
Tools
Safety layers
        ↓
       LLM
        ↓
     Response
```

So:

```text
LLM ≠ complete chatbot product
```

The LLM is the core language model inside a larger system.

---

# 9. Why a normal LLM is not enough

The lecture highlights several limitations of using an LLM by itself.

## Static knowledge

The model's internal parameters come from training. It does not automatically know a newly created private document or a database record just because that information exists somewhere.

## Hallucination

The model can produce fluent text that is incorrect or unsupported.

## Context-window limits

Only a finite amount of text can be supplied to the model at one time.

Dumping an entire company document collection into every prompt is therefore not a good retrieval strategy.

## Domain-knowledge gaps

A general model may not know the detailed information inside:

- university documents,
- internal company policies,
- private manuals,
- recent reports,
- or niche technical documentation.

## Source attribution

A normal generated answer does not automatically tell us which private document supplied the information.

These problems motivate **Retrieval-Augmented Generation**.

---

# 10. What is RAG?

RAG stands for:

```text
Retrieval
Augmented
Generation
```

The central idea is simple:

> **Retrieve useful external information first, then give that information to the LLM as context before asking it to generate the answer.**

A normal LLM flow is roughly:

```text
Question
   ↓
LLM
   ↓
Answer
```

A RAG flow is:

```text
Question
   ↓
Retrieve relevant information
   ↓
Add information to prompt
   ↓
LLM
   ↓
Contextual answer
```

The LLM is not magically searching the vector database by itself.

The retrieval system performs the search first.

---

# 11. RAG versus fine-tuning

Both can adapt an LLM to a domain, but they change different things.

## RAG

```text
Keep model parameters unchanged
       +
Retrieve external information at query time
```

Useful when knowledge changes frequently or comes from documents/databases.

## Fine-tuning

```text
Domain examples
     ↓
Further training
     ↓
Model parameters change
```

Fine-tuning changes the model itself.

### Memory rule

```text
RAG
→ change the CONTEXT

Fine-tuning
→ change the MODEL
```

They are not mutually exclusive; a system can use both.

---

# 12. Complete RAG architecture

The lecture workflow can be divided into two phases.

## Phase A - indexing / preparation

```text
Documents
   ↓
CHUNK
   ↓
EMBED
   ↓
STORE IN VECTOR DATABASE
```

## Phase B - answering a question

```text
QUESTION
   ↓
EMBED QUESTION
   ↓
SEMANTIC SEARCH
   ↓
RETRIEVE TOP CHUNKS
   ↓
CONTEXT
   ↓
PROMPT
   ↓
LLM
   ↓
ANSWER
```

Our main memory chain is:

```text
CHUNK → EMBED → STORE → QUESTION → EMBED → RETRIEVE → CONTEXT → PROMPT → LLM
```

---

# 13. Step 1 - documents

RAG starts with an external knowledge source.

Examples:

```text
PDF
Word document
TXT file
Database
Company documentation
Course material
```

The course exercise reads TXT, PDF, and Word files from a local `docs/` folder.

The important idea is not the file format itself.

The goal is:

> **Extract the text that will become the knowledge base.**

---

# 14. Step 2 - chunking

Large documents are split into smaller pieces called **chunks**.

Why?

Suppose we stored one 200-page PDF as one giant unit.

A question about one paragraph would retrieve the entire document, making retrieval coarse and wasting context-window space.

Instead:

```text
Large document
     ↓
Chunk 1
Chunk 2
Chunk 3
...
Chunk n
```

The lecture highlights two challenges:

- preserve enough context,
- avoid breaking meaningful semantic units unnecessarily.

The exercise uses:

```python
RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
```

`chunk_overlap=50` allows neighbouring chunks to share some text so that information around a boundary is not completely lost.

### Exercise output

The saved exercise produced:

```text
Loaded 13663 text chunks.
```

The exact number depends on the documents in the `docs/` folder.

---

# 15. Metadata

A useful chunk can contain more than text.

Conceptually:

```text
Chunk text
+
metadata
```

Metadata may include:

```text
document title
source
page number
section
date
tags
document ID
```

This helps with filtering, traceability, and returning the original source later.

In the course exercise, the minimal metadata stored with each vector is:

```python
{"text": chunk}
```

That is why the original chunk can be recovered after vector search.

---

# 16. Step 3 - embeddings

An embedding converts text into a numerical vector that represents semantic information.

```text
"machine learning course"
          ↓
Embedding model
          ↓
[0.12, -0.43, 0.81, ...]
```

Unlike one-hot encoding, useful embedding spaces try to place semantically related text closer together.

RAG usually uses **sentence/text embeddings**, because we want to compare the meaning of chunks and questions.

The exercise uses:

```python
SentenceTransformer("all-MiniLM-L6-v2")
```

This model produces a **384-dimensional embedding** in the exercise.

```text
Text chunk
   ↓
all-MiniLM-L6-v2
   ↓
384 numerical values
```

---

# 17. Step 4 - vector database

After creating embeddings, we need to store and search them efficiently.

That is the job of a vector store / vector database.

The lecture mentions tools such as:

- FAISS,
- Pinecone,
- Weaviate,
- Qdrant,
- Milvus,
- Chroma.

The course exercise uses **Pinecone**.

A record conceptually contains:

```python
{
    "id": "42",
    "values": [0.12, -0.43, ...],
    "metadata": {
        "text": "original chunk text"
    }
}
```

Three pieces matter:

```text
ID
→ identifies the record

values
→ embedding used for similarity search

metadata
→ original text / source information
```

The exercise configures the Pinecone index with:

```text
dimension = 384
metric = cosine
```

because the embedding model produces 384-dimensional vectors.

---

# 18. Upsert

The exercise stores records using:

```python
index.upsert(batch)
```

`upsert` means conceptually:

```text
Record already exists
→ update it

Record does not exist
→ insert it
```

The exercise uploads the vectors in batches of 50.

### Saved exercise output

```text
Inserted 13663 vectors into 'rag-tutorial'.
```

---

# 19. Step 5 - embed the question

The documents and the question must live in the **same embedding space**.

So the user's question is passed through the same embedding model.

```python
question_vector = embedder.encode(question).tolist()
```

Now we have:

```text
Document chunks → vectors
Question        → vector
```

This allows mathematical comparison between the question and stored chunks.

---

# 20. Semantic search

Traditional keyword matching asks:

```text
Do the same words appear?
```

Semantic search asks:

```text
Are the meanings similar?
```

The vector database compares the question embedding with stored embeddings and returns the nearest / most similar records.

The exercise retrieves:

```python
results = index.query(
    vector=question_vector,
    top_k=10,
    include_metadata=True,
)
```

Meaning:

```text
vector=question_vector
→ search using the question meaning

top_k=10
→ return the 10 best matches

include_metadata=True
→ also return the stored original text
```

---

# 21. Distance / similarity metrics

Because embeddings are vectors, similarity can be measured mathematically.

The lecture discusses:

## Cosine similarity / distance

Focuses on the angle/direction between vectors.

```text
Similar direction
→ semantically similar
```

## Dot product

Measures vector alignment.

## Euclidean distance (L2)

Measures straight-line distance between vectors.

## Manhattan distance (L1)

Uses the sum of absolute coordinate differences.

The course exercise uses **cosine** similarity in Pinecone.

---

# 22. Step 6 - retrieve the original text

The vector search returns matching records.

The exercise then extracts the text stored in metadata:

```python
contexts = [
    match["metadata"]["text"]
    for match in results["matches"]
]
```

This distinction is extremely important:

```text
Embedding vector
→ used to FIND the chunk

Metadata text
→ used to GIVE the actual information to the LLM
```

The LLM does not need to read a list of 384 floating-point numbers.

It needs the retrieved human-readable text.

---

# 23. Step 7 - augmentation

This is the **A** in RAG.

The retrieved chunks are inserted into the prompt as context.

```python
prompt = f"""Answer the question based only on the context below.

Context:
{context_text}

Question: {question}
Answer:"""
```

So:

```text
Question
+
Retrieved context
+
Instruction
        ↓
Augmented prompt
```

Important:

> **Retrieval happens before generation.**

---

# 24. Step 8 - generation

The exercise uses a local Llama model through Ollama:

```python
from langchain_ollama import OllamaLLM

model = OllamaLLM(model="llama3.2")
```

Then:

```python
response = model.invoke(prompt)
```

The complete last stage is:

```text
Retrieved chunks
      +
User question
      +
Instruction
      ↓
Prompt
      ↓
LLM
      ↓
Generated contextual answer
```

---

# 25. Exercise example and output

The saved notebook asks:

```text
Who is the professor of the course
Climate Change & Environmental Impacts?
```

The retrieved context contains the module's lecturer information, including **Prof. Dr. Oleg Panferov** and **Prof. Dr. Elke Hietel**.

The generated response returned those lecturer names, but it also added unnecessary uncertainty and mentioned information from another retrieved module.

This is actually a useful lesson:

> **RAG improves grounding, but retrieval and generation can still be imperfect.**

Possible problems include:

```text
Irrelevant chunks retrieved
        ↓
Noisy context
        ↓
LLM mixes information
        ↓
Imperfect answer
```

So RAG reduces some hallucination risk; it does not magically guarantee correctness.

---

# 26. What each RAG component is responsible for

| Component | Job |
|---|---|
| Document loader | Extract text from source files |
| Chunker | Break large text into retrievable pieces |
| Embedding model | Convert text into semantic vectors |
| Vector database | Store and search embeddings |
| Question embedding | Represent the user's query in the same vector space |
| Retriever | Select the most relevant chunks |
| Metadata | Bring back the original text/source information |
| Prompt | Combine instruction + context + question |
| LLM | Generate the final natural-language answer |

This table is a very useful debugging tool.

For example:

```text
Wrong chunks retrieved?
→ retrieval / embeddings / chunking problem

Correct chunks retrieved but bad answer?
→ prompt / generation problem
```

---

# 27. RAG is not model training

A common confusion is:

> Do the retrieved documents get trained into the LLM?

No.

In normal RAG:

```text
LLM weights
→ unchanged

Documents
→ stored externally

Retrieved text
→ temporarily supplied in prompt/context
```

The LLM uses the information during that request, but RAG itself does not rewrite the model's learned weights.

---

# 28. Full practical pipeline

```text
DOCS
 ↓
Extract text
 ↓
Chunk documents
 ↓
Embed every chunk
 ↓
Store vector + metadata in Pinecone
 ↓

USER QUESTION
 ↓
Embed question
 ↓
Search Pinecone
 ↓
Retrieve top-k chunks
 ↓
Extract original text from metadata
 ↓
Build augmented prompt
 ↓
Send prompt to Ollama / Llama
 ↓
ANSWER
```

---

# 29. Compact code cheat sheet

```python
# CHUNK
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_text(" ".join(docs))

# EMBEDDING MODEL
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# STORE
vectors = [
    {
        "id": str(i),
        "values": embedder.encode(chunk).tolist(),
        "metadata": {"text": chunk},
    }
    for i, chunk in enumerate(chunks)
]
index.upsert(vectors)

# LLM
model = OllamaLLM(model="llama3.2")

# QUESTION → EMBEDDING
question_vector = embedder.encode(question).tolist()

# RETRIEVE
results = index.query(
    vector=question_vector,
    top_k=10,
    include_metadata=True,
)

# CONTEXT
contexts = [
    match["metadata"]["text"]
    for match in results["matches"]
]
context_text = "\n\n".join(contexts)

# PROMPT
prompt = f"""Answer the question based only on the context below.

Context:
{context_text}

Question: {question}
Answer:"""

# GENERATE
response = model.invoke(prompt)
```

---

# 30. Exam memory block

## LLM

```text
TEXT
→ TOKENS
→ MODEL
→ PROBABILITY FOR EACH NEXT TOKEN
→ SELECT TOKEN
→ APPEND TOKEN
→ REPEAT
```

## RAG

```text
CHUNK
→ EMBED
→ STORE
→ QUESTION
→ EMBED
→ RETRIEVE
→ CONTEXT
→ PROMPT
→ LLM
```

## One-line distinctions

```text
LLM
→ predicts the next token from context

Embedding
→ converts meaning into a vector

Vector DB
→ searches for semantically similar vectors

RAG
→ retrieves external knowledge and adds it to the LLM context

Fine-tuning
→ changes model parameters

RAG
→ changes supplied context, not model parameters
```

---

# 31. Running the practical code

Install the Python dependencies used by the repository version:

```bash
pip install pymupdf python-docx sentence-transformers pinecone langchain-ollama langchain-text-splitters
```

Make sure Ollama is installed and the model is available locally:

```bash
ollama pull llama3.2
```

Set the Pinecone API key as an environment variable rather than committing a secret to GitHub:

```bash
export PINECONE_API_KEY="your-key"
```

Create a `docs/` directory beside `rag_pipeline.py` and place TXT, PDF, or DOCX files inside it.

Then run:

```bash
python rag_pipeline.py
```

The course notebook used a local API-key text file. This repository version deliberately uses an environment variable so the key is not accidentally committed.

---

# Final memory rule

```text
An LLM answers from its current context.

RAG improves that context by retrieving relevant external information first.
```
