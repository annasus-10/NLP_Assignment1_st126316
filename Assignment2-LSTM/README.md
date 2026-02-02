# A2 – Language Modeling with a From-Scratch LSTM

## Overview
This project implements a **character-level language model** trained on a large narrative text corpus using a **Long Short-Term Memory (LSTM) network implemented entirely from scratch**.

The trained model is deployed through a lightweight **Flask web application** that allows users to provide a text prompt and receive an auto-generated continuation.

The project is divided into three main parts:
1. Dataset acquisition and preprocessing  
2. From-scratch LSTM language model training  
3. Web-based inference application  

---

## Dataset

### Source
The dataset used is the **Children Stories Text Corpus**, a collection of classic fairy tales and narrative stories in plain text format.

- Source: Kaggle – *Children Stories Text Corpus*
- Format: Cleaned `.txt` file
- Total characters: ~20 million
- Domain: Narrative / story-like text

This dataset was chosen because:
- it is large enough to train a language model,
- it contains realistic narrative structure (dialogue, punctuation, paragraphing),
- it does not require heavy preprocessing.

### Preprocessing
The model operates at the **character level**, so preprocessing is intentionally minimal:

- The entire text file is read as a single string.
- A vocabulary is built from the **set of all unique characters** in the corpus.
- Two mappings are created:
  - `stoi`: character → index
  - `itos`: index → character
- The full text is converted into a sequence of integer character IDs.

---

## Language Modeling Setup

### Task Definition
Given a sequence of characters:

```

x₁, x₂, …, xₜ

```

the model learns to predict the next character:

```

xₜ₊₁

```

Training is done using fixed-length sliding windows sampled randomly from the corpus.

---

## Model Architecture (From Scratch)

### LSTM Cell
A custom `LSTMCellFromScratch` is implemented using only:
- matrix multiplications,
- elementwise operations,
- sigmoid and tanh activations.

The four standard LSTM gates are explicitly defined:
- input gate
- forget gate
- candidate gate
- output gate

All parameters (`W`, `U`, `b`) are manually created and updated through backpropagation.

### Language Model
The full model consists of:
- Character embedding layer
- Single custom LSTM cell unrolled over time
- Linear output layer projecting hidden states to vocabulary logits

The model predicts a probability distribution over the next character at each time step.

---

## Training Details

- Device: Apple Silicon MPS / CPU fallback
- Embedding dimension: 64
- Hidden dimension: 256
- Sequence length: 128
- Batch size: 64
- Optimizer: Adam
- Training steps: 20,000
- Final training loss: ~1.25

Training is performed using randomly sampled subsequences to efficiently utilize the full dataset without loading all sequences into memory.

---

## Text Generation

### Decoding Strategy
Pure greedy decoding was found to cause repetitive loops.  
Final generation uses:

- **Top-k sampling**
- **Temperature scaling**
- **Repetition penalty** over a sliding character window

This balances coherence and diversity while avoiding degenerate repetition.

---

## Web Application

### Description
A minimal Flask web application is provided to demonstrate inference with the trained model.

Features:
- Single-file `app.py`
- No external templates or assets
- Text prompt input
- Generated continuation output
- Loads trained model and vocabulary artifacts

### How It Works
1. User submits a prompt via a web form.
2. Prompt characters are converted to IDs using `stoi`.
3. The LSTM is run autoregressively to generate new characters.
4. Generated IDs are converted back to text using `itos`.
5. Output is displayed in the browser.

### Running the App
```bash
python app/app.py
````

Then open:

```
http://127.0.0.1:5000
```

![Web App Home Page](web_app_home_page.png)

---

## Project Structure

```
.
├── A2.ipynb                  # Full training and analysis notebook
├── artefacts/
│   ├── story_lstm_scratch_full.pt
│   └── vocab_char_full.json
├── app/
│   └── app.py                # Flask application
├── README.md
```

---

## Reproducibility

* Random seeds are fixed during training.
* Vocabulary is saved explicitly.
* Model weights are saved using `state_dict`.
* The web app reconstructs the model architecture exactly before loading weights.

---

## Notes

* This is a **character-level** model; occasional spelling artifacts are expected.
* Training longer or using word-level modeling would further improve fluency.

---

## Conclusion

This project demonstrates a complete language modeling pipeline using a **from-scratch LSTM**, trained on a realistic narrative dataset and deployed via a simple web interface. All core recurrent computations are implemented manually, satisfying the assignment constraints while producing coherent and meaningful generated text.

