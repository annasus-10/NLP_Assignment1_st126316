# app/app.py
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from flask import Flask, request

# ----------------------------
# Device (Mac M-series uses MPS)
# ----------------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ----------------------------
# From-scratch LSTM (same as notebook)
# ----------------------------
class LSTMCellFromScratch(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.U_i = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.U_f = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.U_g = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.U_o = nn.Parameter(torch.empty(input_dim, hidden_dim))

        self.W_i = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.W_f = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.W_g = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.W_o = nn.Parameter(torch.empty(hidden_dim, hidden_dim))

        self.b_i = nn.Parameter(torch.empty(hidden_dim))
        self.b_f = nn.Parameter(torch.empty(hidden_dim))
        self.b_g = nn.Parameter(torch.empty(hidden_dim))
        self.b_o = nn.Parameter(torch.empty(hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for p in self.parameters():
            p.data.uniform_(-stdv, stdv)

    def forward(self, x_t, state):
        h_t, c_t = state
        f = torch.sigmoid(h_t @ self.W_f + x_t @ self.U_f + self.b_f)
        i = torch.sigmoid(h_t @ self.W_i + x_t @ self.U_i + self.b_i)
        g = torch.tanh   (h_t @ self.W_g + x_t @ self.U_g + self.b_g)
        o = torch.sigmoid(h_t @ self.W_o + x_t @ self.U_o + self.b_o)
        c_new = f * c_t + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class LSTMLanguageModelScratch(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.cell = LSTMCellFromScratch(emb_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def init_state(self, batch_size: int, device):
        h0 = torch.zeros(batch_size, self.cell.hidden_dim, device=device)
        c0 = torch.zeros(batch_size, self.cell.hidden_dim, device=device)
        return (h0, c0)

    def forward(self, x, state=None):
        B, T = x.shape
        if state is None:
            state = self.init_state(B, x.device)

        xemb = self.emb(x)  # (B,T,E)
        h, c = state

        logits_out = []
        for t in range(T):
            h, c = self.cell(xemb[:, t, :], (h, c))
            logits_out.append(self.fc(h).unsqueeze(1))  # (B,1,V)

        logits = torch.cat(logits_out, dim=1)  # (B,T,V)
        return logits, (h, c)


# ----------------------------
# Load artifacts
# ----------------------------
ROOT = Path(__file__).resolve().parent.parent
ART_DIR = ROOT / "artefacts"

VOCAB_PATH = ART_DIR / "vocab_char_full.json"
MODEL_PATH = ART_DIR / "story_lstm_scratch_full.pt"

if not VOCAB_PATH.exists():
    raise FileNotFoundError(f"Missing vocab file: {VOCAB_PATH}")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    itos = json.load(f)["itos"]
stoi = {ch: i for i, ch in enumerate(itos)}
VOCAB_SIZE = len(itos)

model = LSTMLanguageModelScratch(vocab_size=VOCAB_SIZE, emb_dim=64, hidden_dim=256).to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()


# ----------------------------
# Generation (top-k + repetition penalty)
# ----------------------------
@torch.no_grad()
def generate(prompt: str,
             max_new_chars: int = 400,
             temperature: float = 0.45,
             top_k: int = 12,
             repeat_window: int = 200,
             repeat_penalty: float = 1.2) -> str:
    prompt_ids = [stoi[c] for c in prompt if c in stoi]
    if not prompt_ids:
        prompt_ids = [stoi[" "]] if " " in stoi else [0]

    x = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    _, state = model(x, state=None)

    last_id = x[:, -1:]
    out_ids = prompt_ids.copy()

    for _ in range(max_new_chars):
        logits, state = model(last_id, state=state)
        logits = logits[:, -1, :]  # (1,V)

        # repetition penalty on recently used chars
        recent = out_ids[-repeat_window:]
        if recent:
            recent_ids = torch.tensor(list(set(recent)), device=DEVICE, dtype=torch.long)
            logits[0, recent_ids] = logits[0, recent_ids] / repeat_penalty

        logits = logits / max(temperature, 1e-8)

        v, ix = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
        probs = F.softmax(v, dim=-1)
        next_in_topk = torch.multinomial(probs, num_samples=1)
        next_id = ix.gather(-1, next_in_topk)  # (1,1)

        out_ids.append(int(next_id.item()))
        last_id = next_id

    return "".join(itos[i] for i in out_ids)


# ----------------------------
# Minimal UI (single file, no templates)
# ----------------------------
HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Story Generator (From-scratch LSTM)</title>
  <style>
    :root { color-scheme: light; }
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
           max-width: 900px; margin: 40px auto; padding: 0 16px; line-height: 1.4; }
    .card { border: 1px solid #e5e7eb; border-radius: 16px; padding: 18px; background: #fff; }
    textarea { width: 100%; height: 110px; padding: 12px; border-radius: 12px;
               border: 1px solid #d1d5db; font-size: 14px; }
    .row { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-top: 12px; }
    button { padding: 10px 14px; border-radius: 12px; border: 1px solid #111827;
             background: #111827; color: #fff; cursor: pointer; }
    button:hover { opacity: 0.92; }
    .pill { font-size: 12px; padding: 6px 10px; border-radius: 999px; background: #f3f4f6;
            border: 1px solid #e5e7eb; }
    pre { white-space: pre-wrap; background: #0b1220; color: #e5e7eb; padding: 14px;
          border-radius: 16px; overflow: auto; }
    .muted { color: #6b7280; font-size: 12px; }
    a { color: inherit; }
    .grid { display: grid; grid-template-columns: 1fr; gap: 14px; }
  </style>
</head>
<body>
  <h1>Story Generator <span class="muted">(From-scratch LSTM)</span></h1>
  <p class="muted">
    Type a prompt and generate a continuation using a custom LSTM.
  </p>

  <div class="grid">
    <div class="card">
      <form method="POST">
        <label><b>Prompt</b></label>
        <textarea name="prompt" placeholder="Once upon a time...">%PROMPT%</textarea>
        <div class="row">
          <button type="submit">Generate</button>
          <span class="pill">device: %DEVICE%</span>
          <span class="pill">vocab: %VOCAB%</span>
        </div>
        <div class="muted" style="margin-top:8px;">
          Decoding: top-k + repetition penalty (temp=0.45, top_k=12).
        </div>
      </form>
    </div>

    %OUTPUT_BLOCK%
  </div>
</body>
</html>
"""

def render_page(prompt: str, output: str):
    output_block = ""
    if output:
        safe = (output
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))
        output_block = f"""
        <div class="card">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <h2 style="margin:0;">Generated text</h2>
          </div>
          <pre>{safe}</pre>
        </div>
        """
    page = (HTML
            .replace("%PROMPT%", prompt.replace("<", "&lt;").replace(">", "&gt;"))
            .replace("%OUTPUT_BLOCK%", output_block)
            .replace("%DEVICE%", str(DEVICE))
            .replace("%VOCAB%", str(VOCAB_SIZE)))
    return page


# ----------------------------
# Flask routes
# ----------------------------
app = Flask(__name__)

@app.get("/")
def home():
    return render_page(prompt="", output="")

@app.post("/")
def gen():
    prompt = request.form.get("prompt", "")
    output = generate(prompt)
    return render_page(prompt=prompt, output=output)


if __name__ == "__main__":
    # Run: python app/app.py
    app.run(host="127.0.0.1", port=5000, debug=True)
