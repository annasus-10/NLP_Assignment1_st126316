import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, render_template_string

# Optional but recommended: use your myTokenize syllable tokenizer
from myTokenize import SyllableTokenizer

# -------------------------
# Config
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# These filenames must exist in the same folder as this app.py
BASE_DIR = os.path.dirname(__file__)
VOCAB_PATH = os.path.join(BASE_DIR, "..", "artifacts", "vocabs.pt")
MODEL_PATH = os.path.join(BASE_DIR, "..", "artifacts", "best_additive.pt")

# Must match what you trained with (recommended: 128/128 if you used the "optimized" notebook)
EMB_DIM = 128
HID_DIM = 128
DROPOUT = 0.2

PAD = "<pad>"
SOS = "<sos>"
EOS = "<eos>"
UNK = "<unk>"


# -------------------------
# Tokenizers
# -------------------------
mm_tok = SyllableTokenizer()

_en_punct = re.compile(r"([?.!,¿])")
def tokenize_en(text: str):
    text = text.strip().lower()
    text = _en_punct.sub(r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text.split()

def tokenize_mm(text: str):
    return mm_tok.tokenize(text.strip())


# -------------------------
# Model components
# -------------------------
class AdditiveAttention(nn.Module):
    # e_i = v^T tanh(W1 h_i + W2 s)
    def __init__(self, hid_dim):
        super().__init__()
        self.W1 = nn.Linear(hid_dim, hid_dim, bias=False)
        self.W2 = nn.Linear(hid_dim, hid_dim, bias=False)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs, src_mask):
        # decoder_state: (bs,hid)
        # encoder_outputs: (bs,src_len,hid)
        dec = self.W2(decoder_state).unsqueeze(1)  # (bs,1,hid)
        energy = torch.tanh(self.W1(encoder_outputs) + dec)  # (bs,src_len,hid)
        scores = self.v(energy).squeeze(2)  # (bs,src_len)
        scores = scores.masked_fill(~src_mask, -1e9)
        attn = F.softmax(scores, dim=1)  # (bs,src_len)
        context = torch.bmm(attn.unsqueeze(1), encoder_outputs).squeeze(1)  # (bs,hid)
        return context, attn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, emb_dim, hid_dim, pad_idx, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(src_vocab_size, emb_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True)

    def forward(self, src_ids, src_lens):
        emb = self.dropout(self.embedding(src_ids))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, src_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (h, c) = self.lstm(packed)
        enc_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return enc_out, (h, c)


class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, emb_dim, hid_dim, pad_idx, attention, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(tgt_vocab_size, emb_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim + hid_dim, hid_dim, batch_first=True)
        self.attention = attention
        self.fc_out = nn.Linear(hid_dim + hid_dim, tgt_vocab_size)

    def forward(self, input_token, hidden, cell, encoder_outputs, src_mask):
        # input_token: (bs,)
        emb = self.dropout(self.embedding(input_token)).unsqueeze(1)  # (bs,1,emb)
        dec_state = hidden[-1]  # (bs,hid)
        context, attn = self.attention(dec_state, encoder_outputs, src_mask)
        context = context.unsqueeze(1)  # (bs,1,hid)

        lstm_in = torch.cat([emb, context], dim=2)  # (bs,1,emb+hid)
        out, (hidden, cell) = self.lstm(lstm_in, (hidden, cell))  # out: (bs,1,hid)

        out = out.squeeze(1)          # (bs,hid)
        context = context.squeeze(1)  # (bs,hid)

        logits = self.fc_out(torch.cat([out, context], dim=1))  # (bs,vocab)
        return logits, hidden, cell, attn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device


# -------------------------
# Load vocab + model
# -------------------------
if not os.path.exists(VOCAB_PATH):
    raise FileNotFoundError(f"Missing vocabs.pt at: {VOCAB_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Missing best_additive.pt at: {MODEL_PATH}")

vocabs = torch.load(VOCAB_PATH, map_location="cpu")
mm2i = vocabs["mm2i"]
en2i = vocabs["en2i"]
i2en = vocabs["i2en"]

SRC_VOCAB = len(mm2i)
TGT_VOCAB = len(en2i)

attn = AdditiveAttention(HID_DIM)
enc = Encoder(SRC_VOCAB, EMB_DIM, HID_DIM, pad_idx=mm2i[PAD], dropout=DROPOUT)
dec = Decoder(TGT_VOCAB, EMB_DIM, HID_DIM, pad_idx=en2i[PAD], attention=attn, dropout=DROPOUT)
model = Seq2Seq(enc, dec, device=DEVICE).to(DEVICE)

state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()


# -------------------------
# Helpers
# -------------------------
def tokens_to_ids(tokens, tok2i):
    return [tok2i.get(t, tok2i[UNK]) for t in tokens]

@torch.no_grad()
def translate_greedy(mm_text: str, max_len: int = 40) -> str:
    mm_tokens = tokenize_mm(mm_text)
    if len(mm_tokens) == 0:
        return ""

    src_ids = tokens_to_ids(mm_tokens + [EOS], mm2i)
    src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)  # (1,src_len)
    src_lens = torch.tensor([len(src_ids)], dtype=torch.long).to(DEVICE)
    src_mask = (src != mm2i[PAD])  # (1,src_len)

    encoder_outputs, (h, c) = model.encoder(src, src_lens)

    cur = torch.tensor([en2i[SOS]], dtype=torch.long).to(DEVICE)
    out_tokens = []
    last_id = None

    for _ in range(max_len):
        logits, h, c, _ = model.decoder(cur, h, c, encoder_outputs, src_mask)

        # small anti-repeat trick
        top2 = logits.topk(2, dim=1).indices.squeeze(0).tolist()
        nxt = top2[0]
        if last_id is not None and nxt == last_id:
            nxt = top2[1]
        last_id = nxt

        if nxt == en2i[EOS]:
            break

        out_tokens.append(i2en.get(nxt, UNK))
        cur = torch.tensor([nxt], dtype=torch.long).to(DEVICE)

    return " ".join(out_tokens)


# -------------------------
# Flask UI (single file)
# -------------------------
HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Myanmar → English (LSTM + Additive Attention)</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 32px; }
    .card { max-width: 900px; padding: 20px; border: 1px solid #ddd; border-radius: 16px; }
    textarea { width: 100%; height: 120px; padding: 12px; border-radius: 12px; border: 1px solid #ccc; font-size: 16px; }
    button { margin-top: 12px; padding: 10px 16px; border-radius: 12px; border: 1px solid #333; background: #111; color: white; cursor: pointer; }
    .out { margin-top: 16px; padding: 12px; border-radius: 12px; background: #f6f6f6; border: 1px solid #e5e5e5; white-space: pre-wrap; }
    .small { color: #666; font-size: 13px; margin-top: 10px; }
  </style>
</head>
<body>
  <div class="card">
    <h2>Myanmar → English Translation</h2>
    <div class="small">Model: Seq2Seq LSTM + Additive Attention (demo)</div>

    <form method="post">
      <label for="mm_text"><b>Input (Myanmar)</b></label><br/>
      <textarea name="mm_text" id="mm_text" placeholder="Type Myanmar text here...">{{ mm_text }}</textarea><br/>
      <button type="submit">Translate</button>
    </form>

    <h3>Output (English)</h3>
    <div class="out">{{ translation }}</div>

    <div class="small">
      Note: Quality depends on training time and data size. This app demonstrates model integration.
    </div>
  </div>
</body>
</html>
"""

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    mm_text = ""
    translation = ""
    if request.method == "POST":
        mm_text = request.form.get("mm_text", "")
        translation = translate_greedy(mm_text)
    return render_template_string(HTML, mm_text=mm_text, translation=translation)

if __name__ == "__main__":
    # Run: python app.py
    app.run(host="127.0.0.1", port=5000, debug=True)