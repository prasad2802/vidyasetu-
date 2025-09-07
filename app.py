# app.py
import time
import gradio as gr
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import load_baseline, load_corpus, Retriever

# === Load learner model (for adaptation) ===
model, FEATS, val_auc, val_acc = load_baseline()

# === Load generator (explanations) ===
tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
gen = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# === Load retrieval corpus from content/ ===
docs = load_corpus("content")
retr = Retriever(docs)

# === Session learner state (simple) ===
STATE = {"turns":0, "hints":0, "seq_index":0, "rolling_acc_user":0.6, "rolling_acc_skill":0.6, "prev_correct_user":0, "prev_correct_skill":0, "delta_t":0.0}

def adapt_mode(p_next_correct):
    if p_next_correct < 0.5 or STATE["hints"] >= 2:
        return "scaffold"
    if p_next_correct > 0.85 and STATE["hints"] == 0:
        return "challenge"
    return "normal"

def predict_next_prob():
    x = np.array([[
        STATE["seq_index"],
        STATE["prev_correct_user"],
        STATE["rolling_acc_user"],
        STATE["prev_correct_skill"],
        STATE["rolling_acc_skill"],
        STATE["delta_t"]
    ]], dtype=float)
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(x)[0,1]
    else:
        p = 0.6
    return float(p)

def tutor(user_question, want_hint=False):
    t0 = time.time()
    # 1) Retrieve passages
    top = retr.query(user_question, k=3)
    passages = "\n- ".join([d["text"] for d in top]) if top else "No passages available."

    # 2) Predict learning state & choose mode
    p_next = predict_next_prob()
    mode = adapt_mode(p_next)
    style = {
        "scaffold": "Explain step-by-step in simple words. Include a tiny practice question at the end.",
        "challenge": "Give a rigorous explanation and add a tougher follow-up question.",
        "normal": "Explain clearly in 2-3 steps with a small example."
    }[mode]

    # 3) Build prompt
    system = (
        "You are a helpful tutor. Use ONLY the provided passages for accuracy. "
        "If unsure, say you are unsure and suggest checking the textbook.\n"
        f"Passages:\n- {passages}\n\n"
    )
    if want_hint:
        prompt = system + "Give a HINT only (no final answer). Keep it concise and guiding."
        STATE["hints"] += 1
    else:
        prompt = system + f"{style}\nUser question: {user_question}"

    # 4) Generate
    inp = tok(prompt, return_tensors="pt")
    out = gen.generate(**inp, max_new_tokens=220)
    text = tok.decode(out[0], skip_special_tokens=True)

    # 5) Update simple state for next turn
    STATE["turns"] += 1
    STATE["seq_index"] += 1
    # naive update: assume user likely got it right if not asking for a hint
    got_it = 1 if not want_hint else 0
    STATE["prev_correct_user"] = got_it
    STATE["prev_correct_skill"] = got_it
    # rolling averages
    STATE["rolling_acc_user"] = 0.7*STATE["rolling_acc_user"] + 0.3*got_it
    STATE["rolling_acc_skill"] = 0.7*STATE["rolling_acc_skill"] + 0.3*got_it
    STATE["delta_t"] = time.time() - t0

    meta = f"Mode={mode} | p_next={p_next:.2f} | turns={STATE['turns']} | hints={STATE['hints']}"
    return text, meta

with gr.Blocks() as demo:
    gr.Markdown("# Adaptive Tutor Chatbot (ASSISTments)")
    if val_auc is not None and val_acc is not None:
        gr.Markdown(f"**Baseline model:** Val AUC={val_auc:.3f}, Acc={val_acc:.3f}")
    q = gr.Textbox(label="Ask a question from your syllabus")
    hint = gr.Checkbox(label="Hint only")
    a = gr.Textbox(label="Tutor")
    meta = gr.Textbox(label="State")
    btn = gr.Button("Send")
    btn.click(tutor, [q, hint], [a, meta])

demo.queue().launch()
