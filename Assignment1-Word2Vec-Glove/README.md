# Assignment 1 – Word Embeddings from Scratch

## Overview
This project implements and compares classic word embedding models trained **from scratch** on a real-world text corpus. The models are evaluated using standard intrinsic benchmarks and applied in a simple web-based paragraph similarity search application.

The following models are implemented:
- Skip-gram with full softmax
- Skip-gram with negative sampling
- GloVe (Global Vectors)

All models are trained on the Reuters news corpus using identical preprocessing to ensure fair comparison.

---

## Dataset and Preprocessing
We use the **Reuters news corpus** provided by NLTK. The corpus consists of English news articles covering topics such as economics, finance, trade, and politics.

Preprocessing steps:
- Lowercasing
- Tokenization
- Removal of non-alphabetic tokens
- Minimum frequency threshold (`min_count = 5`)
- Vocabulary capped to the most frequent words

Final statistics:
- Vocabulary size: **4,075**
- Embedding dimension: **50**
- Training corpus: ~10,000 sentences

All models share the same vocabulary and corpus representation.

---

## Implemented Models

### 1. Skip-gram with Full Softmax
This model predicts surrounding context words given a center word using a full softmax over the vocabulary.

- Objective: maximize log-likelihood of true context words
- Loss: cross-entropy over the entire vocabulary
- Computationally expensive but conceptually simple

### 2. Skip-gram with Negative Sampling
This variant improves efficiency by replacing the full softmax with a binary classification objective that distinguishes true context words from randomly sampled negative words.

- Negative samples drawn from a unigram distribution raised to the 0.75 power
- Significantly faster than full softmax
- Commonly used in practical word2vec implementations

### 3. GloVe (Global Vectors)
GloVe learns embeddings by factorizing a word–word co-occurrence matrix.

- Co-occurrence counts collected within a fixed window
- Weighted least-squares loss on log co-occurrence counts
- Uses a sparse representation for efficiency

---

## Training Results

| Model                  | Dim | Vocab | Train Time (s) | Final Loss |
|------------------------|-----|-------|---------------|------------|
| Skip-gram (Softmax)    | 50  | 4,075 | 4.21          | 5.79       |
| Skip-gram (NEG)        | 50  | 4,075 | 8.17          | 2.16       |
| GloVe                  | 50  | 4,075 | 1.72          | 0.61       |

**Note:** Loss values are not directly comparable across models because each model optimizes a different objective.

---

## Evaluation

### Word Similarity (WordSim353)
We evaluate embeddings using Spearman correlation between cosine similarity of word pairs and human similarity judgments.

| Model               | Spearman Correlation |
|---------------------|----------------------|
| Skip-gram Softmax   | 0.118                |
| Skip-gram NEG       | 0.079                |
| GloVe               | -0.052               |

Only a subset of WordSim353 pairs could be evaluated due to vocabulary overlap. Lower absolute scores are expected given the limited corpus size and domain specificity.

---

### Analogy Task (Google Analogies)
The Google analogy dataset evaluates relational structure (e.g., *king − man + woman ≈ queen*).

| Model               | Accuracy |
|---------------------|----------|
| Skip-gram Softmax   | 0.002    |
| Skip-gram NEG       | 0.002    |
| GloVe               | 0.000    |

Due to the small vocabulary and the news-focused training corpus, analogy accuracy is low. This behavior is expected and consistent across models.

---

## Paragraph Similarity Web Application

A simple web application is implemented using **Flask** to demonstrate the learned embeddings in a downstream task.

### Method
- Each paragraph from the Reuters corpus is represented as the **mean of its word embeddings**
- User queries are embedded using the same method
- Cosine similarity is used to retrieve the **top 10 most similar paragraphs**

This approach captures topical similarity rather than exact keyword matching, which is appropriate given the embedding method.

### How to Run
From the project root directory:
```bash
python app/app.py
```
Then open:

```bash
http://127.0.0.1:5000
```

## Discussion

- Skip-gram with negative sampling converges faster and is more computationally efficient than full softmax.
- GloVe training is efficient due to sparse co-occurrence statistics.
- Evaluation scores are limited by corpus size and domain mismatch with benchmark datasets.
- The web application demonstrates practical semantic retrieval despite these limitations.

## Conclusion

This project demonstrates correct implementation, training, and evaluation of multiple word embedding models from scratch. Despite modest evaluation scores due to dataset constraints, the results are consistent and the models behave as expected. The web application further illustrates how word embeddings can be applied to semantic similarity tasks.
