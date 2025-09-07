# app_gpt.py  — Adaptive Tutor that uses GPT API for answers
# Requires: pip install openai gradio joblib

import os, time, joblib
import gradio as gr
import numpy as np
from openai import OpenAI

# ---------- Load the baseline (ASSISTments) model for adaptation ----------
def load_baseline(path="models/baseline.pkl"):
    class FallbackModel:
        def predict_proba(self, X):
            p = np.full((X.shape[0], 2), 0.0)
            p[:,1] = 0.6  # neutral prior
            p[:,0] = 0.4
            return p

    if os.path.exists(path):
        bundle = joblib.load(path)
        return bundle["model"], bundle["feats"], bundle.get("val_auc"), bundle.get("val_acc")
    return FallbackModel(), ["seq_index","prev_correct_user","rolling_acc_user",
                             "prev_correct_skill","rolling_acc_skill","delta_t"], None, None

model, FEATS, val_auc, val_acc = load_baseline()

# ---------- GPT client (key must be in env: OPENAI_API_KEY) ----------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def gpt_tutor_answer(question: str, mode: str) -> str:
    """Call GPT with an instruction tuned to the chosen mode."""
    style = {
        "scaffold":  ("Explain step-by-step in very simple words like teaching a beginner. "
                      "Use short sentences and a numbered list. End with one tiny practice question."),
        "normal":    ("Explain clearly in 2–3 short steps and give one small example."),
        "challenge": ("Give a rigorous explanation and add a harder follow-up question that checks understanding.")
    }[mode]

    prompt = (
        "You are a patient, accurate tutor. Avoid unnecessary jargon. "
        "If the question is purely arithmetic, show the steps and final answer. "
        f"{style}\n\nQuestion: {question}"
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return resp.choices[0].message.content.strip()

# ---------- Minimal learner state to feed the baseline features ----------
STATE = {
    "turns": 0, "hints": 0,
    "seq_index": 0,
    "rolling_acc_user": 0.6, "rolling_acc_skill": 0.6,
    "prev_correct_user": 0,  "prev_correct_skill": 0,
    "delta_t": 0.0,
}

def predict_next_prob():
    x = np.array([[
        STATE["seq_index"],
        STATE["prev_correct_user"],
        STATE["rolling_acc_user"],
        STATE["prev_correct_skill"],
        STATE["rolling_acc_skill"],
        STATE["delta_t"],
    ]], dtype=float)
    p = model.predict_proba(x)[0,1] if hasattr(model, "predict_proba") else 0.6
    return float(p)

def adapt_mode(p_next):
    if p_next < 0.5 or STATE["hints"] >= 2:
        return "scaffold"
    if p_next > 0.85 and STATE["hints"] == 0:
        return "challenge"
    return "normal"

def tutor(user_question: str, hint_only: bool):
    t0 = time.time()

    # 1) Predict next-correctness & choose mode
    p_next = predict_next_prob()
    mode = adapt_mode(p_next)

    # 2) Ask GPT (hint = just ask for a hint)
    mode_for_gpt = "scaffold" if hint_only else mode
    answer = gpt_tutor_answer(user_question, mode_for_gpt)

    # 3) Update simple learner state
    STATE["turns"] += 1
    STATE["seq_index"] += 1
    got_it = 0 if hint_only else 1  # naive: if not asking for hint, treat as progress
    STATE["prev_correct_user"]  = got_it
    STATE["prev_correct_skill"] = got_it
    STATE["rolling_acc_user"]   = 0.7 * STATE["rolling_acc_user"]  + 0.3 * got_it
    STATE["rolling_acc_skill"]  = 0.7 * STATE["rolling_acc_skill"] + 0.3 * got_it
    STATE["delta_t"] = time.time() - t0
    if hint_only:
        STATE["hints"] += 1
    else:
        STATE["hints"] = max(0, STATE["hints"] - 1)  # decay hint pressure

    meta = f"Mode={mode_for_gpt} | p_next={p_next:.2f} | turns={STATE['turns']} | hints={STATE['hints']}"
    return answer, meta

# ---------- Gradio UI ----------
with gr.Blocks() as demo:
    gr.Markdown("# Adaptive Tutor Chatbot (GPT API)\n"
                + (f"**Baseline model:** Val AUC={val_auc:.3f}, Acc={val_acc:.3f}" if val_auc else ""))
    q = gr.Textbox(label="Ask anything (math, science, history, etc.)")
    hint = gr.Checkbox(label="Hint only")
    a = gr.Textbox(label="Tutor")
    meta = gr.Textbox(label="State")
    btn = gr.Button("Send")
    btn.click(fn=tutor, inputs=[q, hint], outputs=[a, meta])

demo.queue().launch()
