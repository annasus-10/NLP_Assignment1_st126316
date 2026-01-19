import json, re
import numpy as np
from flask import Flask, request, render_template_string
from pathlib import Path

# ---------- Paths ----------
APP_DIR = Path(__file__).resolve().parent          # ASSIGNMENT1/app
ROOT_DIR = APP_DIR.parent                          # ASSIGNMENT1
EMB_DIR = ROOT_DIR / "embeddings"                  # ASSIGNMENT1/embeddings

# ---------- Load paragraph data ----------
V = np.load(APP_DIR / "paragraph_vectors.npy")
with open(APP_DIR / "paragraph_texts.json", "r") as f:
    PARAS = json.load(f)

# ---------- Load vocabulary ----------
with open(APP_DIR / "word2id.json", "r") as f:
    word2id = json.load(f)

# ---------- Load embeddings ----------
E = np.load(EMB_DIR / "sg_neg_embeddings.npy")  # or glove_embeddings.npy

# ---------- Helpers ----------
def normalize_rows(M, eps=1e-9):
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    return M / (norms + eps)

E_n = normalize_rows(E)

def text_to_ids(text):
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return [word2id.get(t, 0) for t in tokens]

def embed_query(text):
    ids = [i for i in text_to_ids(text) if i != 0]
    if not ids:
        return None
    q = E_n[ids].mean(axis=0)
    q = q / (np.linalg.norm(q) + 1e-9)
    return q.astype(np.float32)

# ---------- HTML ----------
HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Paragraph Similarity Search</title>
  <style>
    body { font-family: system-ui; margin: 24px; max-width: 980px; }
    textarea { width: 100%; height: 120px; padding: 10px; }
    button { padding: 10px 14px; margin-top: 8px; }
    .result { border: 1px solid #ddd; padding: 12px; margin-top: 10px; border-radius: 10px; }
    .score { font-size: 12px; color: #555; }
    pre { white-space: pre-wrap; }
  </style>
</head>
<body>
<h1>Paragraph Similarity Search (Top 10)</h1>

<form method="POST">
<textarea name="query" placeholder="Paste a paragraph...">{{query}}</textarea><br/>
<button type="submit">Search</button>
</form>

{% if error %}
<p style="color:red;"><b>{{error}}</b></p>
{% endif %}

{% for score, text in results %}
<div class="result">
  <div class="score">cosine similarity: {{ "%.4f"|format(score) }}</div>
  <pre>{{text}}</pre>
</div>
{% endfor %}

</body>
</html>
"""

# ---------- Flask ----------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    results = []
    error = None

    if request.method == "POST":
        query = request.form.get("query", "")
        q = embed_query(query)
        if q is None:
            error = "No in-vocabulary words found."
        else:
            sims = V @ q
            topk = 10
            idx = np.argpartition(-sims, topk)[:topk]
            idx = idx[np.argsort(-sims[idx])]
            results = [(float(sims[i]), PARAS[i]) for i in idx]

    return render_template_string(HTML, query=query, results=results, error=error)

if __name__ == "__main__":
    app.run(debug=True, port=5000)