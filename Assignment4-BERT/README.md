# Assignment 4 – Do You AGREE?

Natural Language Inference using Scratch BERT and Sentence-BERT

## Overview

This project implements a Natural Language Inference (NLI) system consisting of:

1. BERT trained from scratch using Masked Language Modeling (MLM)
2. Sentence-BERT (SBERT) style siamese architecture
3. Softmax classification objective for NLI
4. A simple web application for inference

The system classifies the logical relationship between two sentences as:

* Entailment
* Neutral
* Contradiction

---

# Task 1 – Training BERT from Scratch

## Model Configuration

A custom BERT model was initialized using `BertConfig` with:

* Hidden size: 256
* Number of layers: 4
* Attention heads: 4
* Intermediate size: 1024
* Maximum sequence length: 128

The model was initialized with random weights and trained using Masked Language Modeling.

## Dataset

* Dataset: WikiText-2 (raw)
* Objective: Masked Language Modeling (MLM)
* Epochs: 1
* Batch size: 8
* Optimizer: AdamW
* Learning rate: 5e-4
* Weight decay: 0.01

After training, the model weights were saved and reused for downstream fine-tuning.

---

# Task 2 – Sentence-BERT for NLI

A siamese architecture was constructed where both sentences are encoded independently using the same BERT encoder.

Sentence A → BERT → u
Sentence B → BERT → v

Mean pooling was applied to obtain sentence-level embeddings.

The final feature representation is:

(u, v, |u − v|)

This concatenated vector is passed into a linear classifier:

Linear(3 × hidden_size → 3 classes)

Loss function: CrossEntropyLoss (Softmax classification objective)

## Dataset

* Dataset: SNLI (Stanford Natural Language Inference)
* 20,000 training samples
* 5,000 validation samples
* Labels: entailment, neutral, contradiction

---

# Task 3 – Evaluation

Validation Results:

| Label         | Precision | Recall | F1-score |
| ------------- | --------- | ------ | -------- |
| Entailment    | 0.50      | 0.65   | 0.56     |
| Neutral       | 0.50      | 0.42   | 0.46     |
| Contradiction | 0.47      | 0.40   | 0.43     |

Overall Accuracy: 49%

## Analysis

* Performance is significantly above random baseline (33%).
* Entailment is detected more reliably than neutral or contradiction.
* Limited performance is expected due to:

  * Small model size
  * Limited MLM pretraining (1 epoch)
  * Subset of SNLI used
  * Computational constraints

Potential improvements:

* Increase MLM pretraining epochs
* Increase SBERT fine-tuning epochs
* Train on full SNLI dataset
* Increase hidden size or number of layers
* Apply learning rate scheduling

---

# Task 4 – Web Application

A Streamlit web application was developed with:

* Two input fields (Premise and Hypothesis)
* Prediction of NLI label
* Display of class probabilities
* Custom-trained SBERT model loading

Run with:

```
streamlit run app.py
```

---

# Project Structure

```
Assignment4-BERT/
│
├── notebook.ipynb
├── app.py
├── README.md
├── sbert_nli_model/
│   ├── config.json
│   ├── model weights
│   ├── tokenizer files
│   └── classifier_head.pt
```

---

# How to Run

Install dependencies:

```
pip install torch transformers datasets streamlit scikit-learn
```

1. Run the notebook to train and save the model.
2. Run the web application:

```
streamlit run app.py
```

---

# Conclusion

This project demonstrates:

* BERT pretraining from scratch using MLM
* Sentence-level semantic representation
* Siamese network architecture for NLI
* Softmax classification objective
* End-to-end NLI system deployment
