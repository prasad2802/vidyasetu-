from explainer_groq import explain_with_groq
# app_hf_practice.py — FREE local Hugging Face model for Tutor + Auto-MCQ Practice
# Run:  python app_hf_practice.py
import os, re, time, random, joblib
import gradio as gr
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------- Personalization baseline (from your ASSISTments model if present) ----------
def load_baseline(path="models/baseline.pkl"):
    class FallbackModel:
        def predict_proba(self, X):
            p = np.full((X.shape[0], 2), 0.0); p[:,1]=0.6; p[:,0]=0.4; return p
    if os.path.exists(path):
        b = joblib.load(path); return b["model"], b["feats"], b.get("val_auc"), b.get("val_acc")
    return FallbackModel(), ["seq_index","prev_correct_user","rolling_acc_user",
                             "prev_correct_skill","rolling_acc_skill","delta_t"], None, None

model_base, FEATS, val_auc, val_acc = load_baseline()

# -------- Load FREE local HF model (no API/billing) ----------
MODEL_NAME = os.getenv("HF_MODEL_NAME", "google/flan-t5-base")  # use flan-t5-small for speed if needed
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
hf = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
torch.set_num_threads(max(1, os.cpu_count() // 2))

def generate(txt, max_new=220, temperature=0.3, top_p=0.9):
    inp = tok(txt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = hf.generate(**inp, max_new_tokens=max_new, do_sample=True, temperature=temperature, top_p=top_p)
    return tok.decode(out[0], skip_special_tokens=True).strip()

# -------- Minimal learner state ----------
STATE = {"turns":0,"hints":0,"seq_index":0,"rolling_acc_user":0.6,"rolling_acc_skill":0.6,
         "prev_correct_user":0,"prev_correct_skill":0,"delta_t":0.0}

def predict_next_prob():
    x = np.array([[STATE["seq_index"],STATE["prev_correct_user"],STATE["rolling_acc_user"],
                   STATE["prev_correct_skill"],STATE["rolling_acc_skill"],STATE["delta_t"]]], float)
    return float(model_base.predict_proba(x)[0,1]) if hasattr(model_base,"predict_proba") else 0.6

def adapt_mode(p):
    if p < 0.5 or STATE["hints"] >= 2: return "scaffold"
    if p > 0.85 and STATE["hints"] == 0: return "challenge"
    return "normal"

# -------- Topic notes (seed content for MCQ generation) ----------
TOPICS = {
  "Fractions": """Fractions:
- a/b means 'a' parts out of 'b' equal parts.
- Add: common denominator, then add numerators.
- Multiply: (a/b)*(c/d)=(ac)/(bd). Divide: (a/b)÷(c/d)=(a/b)*(d/c).
- Always simplify the final answer.""",
  "Decimals": """Decimals:
- Line up decimal points for + and −.
- For ×: multiply like integers, then place decimal using total decimal digits.
- For ÷ by decimal: scale both numbers by 10^k to make divisor whole.""",
  "Percents": """Percents:
- x% = x/100. x% of N = (x/100)*N.
- Increase/Decrease: New = Original*(1±r) where r is decimal rate."""
}

# -------- Tutor tab (free-form Q&A with adaptive style) ----------
def tutor(question, want_hint):
    t0 = time.time()
    p = predict_next_prob()
    mode = "scaffold" if want_hint else adapt_mode(p)
    style = {
        "scaffold":"Explain step-by-step in simple words. Use a short numbered list. End with one tiny practice.",
        "normal":"Explain clearly in 2–3 short steps and give one small example.",
        "challenge":"Give a rigorous explanation and add a harder follow-up question."
    }[mode]
    prompt = f"You are a patient school tutor. {style}\nQuestion: {question}\nAnswer:"
    ans = explain_with_groq(prompt)
    # update state
    STATE["turns"] += 1; STATE["seq_index"] += 1
    got_it = 0 if want_hint else 1
    STATE["prev_correct_user"]=got_it; STATE["prev_correct_skill"]=got_it
    STATE["rolling_acc_user"]=0.7*STATE["rolling_acc_user"]+0.3*got_it
    STATE["rolling_acc_skill"]=0.7*STATE["rolling_acc_skill"]+0.3*got_it
    STATE["delta_t"]=time.time()-t0
    STATE["hints"] = STATE["hints"]+1 if want_hint else max(0, STATE["hints"]-1)
    meta = f"Mode={mode} | p_next={p:.2f} | turns={STATE['turns']} | hints={STATE['hints']} | model={MODEL_NAME} | device={device}"
    return ans, meta

# -------- Practice tab (auto-generate MCQs; check answers; adapt) ----------
def make_mcqs(topic_text, n=5):
    """Prompt FLAN-T5 to create n MCQs with exactly 4 options and indicate the correct letter."""
    prompt = f"""Create {n} multiple-choice questions (MCQ) from the topic notes below.
Rules:
- For each MCQ, provide: Q:, A), B), C), D), Correct:, Explanation:
- Exactly one correct option; Correct should be a single letter A/B/C/D.
- Keep questions concise for grade 6.
Topic notes:
{topic_text}"""
    raw = generate(prompt, max_new=400)
    # naive parse
    items, cur = [], {}
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    for ln in lines:
        if ln.startswith("Q:"):
            if cur: items.append(cur); cur={}
            cur = {"Q": ln[2:].strip(), "A":None,"B":None,"C":None,"D":None,"Correct":None,"Expl":""}
        elif ln.startswith("A)"): cur["A"]=ln[2:].strip()
        elif ln.startswith("B)"): cur["B"]=ln[2:].strip()
        elif ln.startswith("C)"): cur["C"]=ln[2:].strip()
        elif ln.startswith("D)"): cur["D"]=ln[2:].strip()
        elif ln.lower().startswith("correct"):
            m = re.search(r"([A-D])", ln.upper()); cur["Correct"] = m.group(1) if m else None
        elif ln.lower().startswith("explanation"):
            cur["Expl"] = ln.split(":",1)[-1].strip()
        else:
            # append to the last field if explanation started
            if "Expl" in cur and cur["Expl"] is not None:
                cur["Expl"] = (cur["Expl"]+" "+ln).strip()
    if cur: items.append(cur)
    # keep only valid MCQs
    clean = [it for it in items if all(it.get(k) for k in ["Q","A","B","C","D","Correct"])]
    return clean[:n] if clean else []

P_SESS = {"topic":"Fractions","pool":[], "idx":0, "corrects":0, "attempts":0}

def start_topic(topic):
    P_SESS["topic"]=topic
    P_SESS["pool"]=make_mcqs(TOPICS[topic], n=5)
    P_SESS["idx"]=0
    P_SESS["corrects"]=0
    P_SESS["attempts"]=0
    if not P_SESS["pool"]:
        return "Failed to auto-generate MCQs. Try again.", "", ""
    q = P_SESS["pool"][0]
    text = f"{q['Q']}\nA){q['A']}\nB){q['B']}\nC){q['C']}\nD){q['D']}"
    return text, "", f"Loaded {len(P_SESS['pool'])} MCQs for {topic}."

def check_answer(choice):
    if not P_SESS["pool"]:
        return "No quiz loaded. Pick a topic first.", "", ""
    q = P_SESS["pool"][P_SESS["idx"]]
    correct = q["Correct"].upper().strip() if q["Correct"] else "A"
    P_SESS["attempts"] += 1
    is_right = (choice.upper().strip()==correct)
    if is_right:
        P_SESS["corrects"] += 1
        feedback = f"✅ Correct ({correct}). {q.get('Expl','')}"
        # advance
        P_SESS["idx"] += 1
        if P_SESS["idx"] >= len(P_SESS["pool"]):
            # Topic complete → suggest next one
            topics = list(TOPICS.keys())
            nxt = topics[(topics.index(P_SESS["topic"])+1)%len(topics)]
            return feedback, "", f"Great! Topic '{P_SESS['topic']}' complete. Try next: {nxt}"
        else:
            nq = P_SESS["pool"][P_SESS["idx"]]
            text = f"{nq['Q']}\nA){nq['A']}\nB){nq['B']}\nC){nq['C']}\nD){nq['D']}"
            return feedback, text, f"Score: {P_SESS['corrects']}/{P_SESS['attempts']}"
    else:
        # wrong → give two more MCQs in same topic if available (regen small set)
        hint_prompt = f"Give a short hint for this question: {q['Q']}.\nKeep it very simple."
        hint = generate(hint_prompt, max_new=80, temperature=0.4)
        # regen one extra mcq to reinforce (optional)
        extra = make_mcqs(TOPICS[P_SESS["topic"]], n=1)
        if extra:
            P_SESS["pool"][P_SESS["idx"]:P_SESS["idx"]] = extra  # insert before current to repeat topic
        text = f"{q['Q']}\nA){q['A']}\nB){q['B']}\nC){q['C']}\nD){q['D']}"
        return f"❌ Incorrect. Correct is {correct}. Hint: {hint}", text, f"Score: {P_SESS['corrects']}/{P_SESS['attempts']} (Reinforcing {P_SESS['topic']})"

# -------- Build UI ----------
with gr.Blocks() as demo:
    gr.Markdown("# Personalized Learning — Tutor + Auto-MCQ (FREE Hugging Face)")
    if val_auc: gr.Markdown(f"**Baseline personalization model:** Val AUC={val_auc:.3f}")

    with gr.Tab("Tutor (Ask Anything)"):
        q = gr.Textbox(label="Your question")
        hint = gr.Checkbox(label="Hint only")
        a = gr.Textbox(label="Tutor answer")
        meta = gr.Textbox(label="State / Debug")
        gr.Button("Send").click(tutor, [q, hint], [a, meta])

    with gr.Tab("Practice (Auto MCQ)"):
        topic = gr.Radio(choices=list(TOPICS.keys()), value="Fractions", label="Choose Topic")
        load_btn = gr.Button("Start Topic")
        mcq = gr.Textbox(label="Question", lines=6)
        choice = gr.Radio(choices=["A","B","C","D"], label="Your Answer")
        submit = gr.Button("Submit Answer")
        feedback = gr.Textbox(label="Feedback / Explanation")
        status = gr.Textbox(label="Progress")

        load_btn.click(start_topic, [topic], [mcq, feedback, status])
        submit.click(check_answer, [choice], [feedback, mcq, status])

demo.queue().launch()
