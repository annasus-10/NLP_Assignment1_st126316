import os
import time
import torch
import torch.nn as nn
import streamlit as st
from transformers import BertModel, BertTokenizerFast

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="NLI Demo (SBERT from Scratch)", page_icon="🧠", layout="centered")

DEFAULT_MODEL_DIR = "./sbert_nli_model"
LABELS = ["entailment", "neutral", "contradiction"]  # SNLI mapping: 0/1/2
MAX_LEN = 128


# ----------------------------
# Device selection
# ----------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------
# SBERT utilities
# ----------------------------
def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden_state: (batch, seq, hidden)
    # attention_mask:    (batch, seq)
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # (batch, seq, 1)
    summed = torch.sum(last_hidden_state * mask, dim=1)            # (batch, hidden)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)                # (batch, 1)
    return summed / counts


def configurations(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # builds (u, v, |u-v|) just like your notebook
    uv_abs = torch.abs(u - v)
    return torch.cat([u, v, uv_abs], dim=-1)


# ----------------------------
# Model loading (cached)
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_dir: str):
    device = get_device()

    # Load tokenizer + encoder
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    encoder = BertModel.from_pretrained(model_dir).to(device)
    encoder.eval()

    # Build classifier head as plain Linear(hidden*3 -> 3)
    hidden = encoder.config.hidden_size
    head = nn.Linear(hidden * 3, 3).to(device)

    head_path = os.path.join(model_dir, "classifier_head.pt")
    if not os.path.exists(head_path):
        raise FileNotFoundError(f"Missing classifier head: {head_path}")

    state = torch.load(head_path, map_location="cpu")  # expects keys: weight, bias
    head.load_state_dict(state)
    head.eval()

    return tokenizer, encoder, head, device


# ----------------------------
# Inference
# ----------------------------
def predict(tokenizer, encoder, head, device, premise: str, hypothesis: str):
    enc_a = tokenizer(
        premise,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    enc_b = tokenizer(
        hypothesis,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )

    input_ids_a = enc_a["input_ids"].to(device)
    attn_a = enc_a["attention_mask"].to(device)
    input_ids_b = enc_b["input_ids"].to(device)
    attn_b = enc_b["attention_mask"].to(device)

    with torch.no_grad():
        out_a = encoder(input_ids_a, attention_mask=attn_a)
        out_b = encoder(input_ids_b, attention_mask=attn_b)

        u = mean_pool(out_a.last_hidden_state, attn_a)
        v = mean_pool(out_b.last_hidden_state, attn_b)

        x = configurations(u, v)
        logits = head(x)

        probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu()
        pred_id = int(torch.argmax(probs).item())

    return pred_id, probs.tolist()


# ----------------------------
# UI
# ----------------------------
st.title("🧠 NLI Demo — SBERT (Scratch BERT + Softmax Classifier)")
st.caption("Enter a premise and a hypothesis. The model predicts entailment / neutral / contradiction.")

with st.expander("⚙️ Model settings", expanded=False):
    model_dir = st.text_input("Model directory", value=DEFAULT_MODEL_DIR)
    st.write("Expected files inside:", "`config.json`, encoder weights, tokenizer files, `classifier_head.pt`")
    st.write("Device auto-selects (MPS/CUDA/CPU).")

premise = st.text_area(
    "Premise",
    value="A man is playing a guitar on stage.",
    height=90,
)

hypothesis = st.text_area(
    "Hypothesis",
    value="The man is performing music.",
    height=90,
)

col1, col2 = st.columns([1, 1])
with col1:
    run_btn = st.button("Predict", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("Clear", use_container_width=True)

if clear_btn:
    st.session_state.clear()
    st.rerun()

# Load model
try:
    with st.spinner("Loading model..."):
        tokenizer, encoder, head, device = load_model(model_dir)
    st.success(f"Loaded model from: {model_dir} | device: {device}")
except Exception as e:
    st.error("Could not load model. Check your model directory and files.")
    st.code(str(e))
    st.stop()

if run_btn:
    if not premise.strip() or not hypothesis.strip():
        st.warning("Please enter both premise and hypothesis.")
        st.stop()

    t0 = time.time()
    pred_id, probs = predict(tokenizer, encoder, head, device, premise, hypothesis)
    dt = (time.time() - t0) * 1000

    st.subheader("Prediction")
    st.write(f"**Label:** `{LABELS[pred_id]}`")
    st.caption(f"Inference time: {dt:.1f} ms")

    st.subheader("Probabilities")
    for label, p in zip(LABELS, probs):
        st.write(f"- **{label}**: {p:.4f}")
