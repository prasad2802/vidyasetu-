# app_groq_practice.py
import os, time, random, requests
os.environ["GRADIO_ROOT_PATH"] = "/tutor"
import gradio as gr
import sympy as sp

TOPIC_TARGET_CORRECT = 5
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_FALLBACKS = [
    os.getenv("GROQ_MODEL", "").strip() or "llama-3.3-70b-versatile",
    "gemma2-9b-it",
    "llama-3.1-8b-instant",
]
_LAST_WORKING_MODEL = None

def explain_with_groq(prompt: str) -> str:
    """Return a string; never raise so the UI can load even if the key is wrong."""
    global _LAST_WORKING_MODEL
    key = (os.getenv("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") or "").strip().strip('"').strip("'")
    if not key.startswith("gsk_"):
        return "❌ GROQ_API_KEY missing/invalid on server."

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try_order = []
    if _LAST_WORKING_MODEL:
        try_order.append(_LAST_WORKING_MODEL)
    for m in GROQ_FALLBACKS:
        if m and m not in try_order:
            try_order.append(m)

    last_err = None
    for model in try_order:
        try:
            r = requests.post(
                GROQ_URL,
                headers=headers,
                json={"model": model, "messages": [{"role":"user","content":prompt}], "temperature": 0.2},
                timeout=60,
            )
            if r.status_code == 200:
                _LAST_WORKING_MODEL = model
                return r.json()["choices"][0]["message"]["content"].strip()
            txt = r.text[:350]
            if r.status_code in (401, 403): return f"❌ Groq {r.status_code} {txt}"
            if r.status_code in (404, 422, 429): 
                last_err = f"ℹ️ {r.status_code} on {model}: {txt}"
                continue
            last_err = f"❌ HTTP {r.status_code} on {model}: {txt}"
        except Exception as e:
            last_err = f"❌ Request failed on {model}: {type(e).__name__}: {e}"
    return last_err or "❌ All Groq model attempts failed."

def solve_math(expr: str):
    try:
        expr = expr.replace("=", "")
        return str(sp.simplify(expr))
    except Exception:
        return None

STATE = {"turns":0, "rolling_acc_user":0.7, "rolling_acc_skill":0.7, "hints":0}

def predict_next_prob():
    return 0.5*STATE["rolling_acc_user"] + 0.5*STATE["rolling_acc_skill"]

def adapt_mode(p):
    if p < 0.4: return "scaffold"
    if p < 0.7: return "normal"
    return "challenge"

def tutor(question, want_hint):
    t0 = time.time()
    p = predict_next_prob()
    mode = "scaffold" if want_hint else adapt_mode(p)
    style = {
        "scaffold": "Explain step-by-step in simple words. Use a short list. End with one tiny practice.",
        "normal":   "Explain clearly in 2–3 steps with a small example.",
        "challenge":"Give a rigorous explanation and add a harder follow-up question."
    }[mode]

    looks_math = any(op in question for op in "+-*/=")
    computed = solve_math(question) if looks_math else None
    if computed is not None:
        prompt = f"You are a math teacher. {style}\nExplain why the result of '{question}' is {computed}."
        ans = f"✅ Answer: {computed}\n" + explain_with_groq(prompt)
    else:
        ans = explain_with_groq(f"You are a patient tutor. {style}\nQuestion: {question}\nAnswer:")

    STATE["turns"] += 1
    got_it = 0 if want_hint else 1
    STATE["rolling_acc_user"]  = 0.7*STATE["rolling_acc_user"]  + 0.3*got_it
    STATE["rolling_acc_skill"] = 0.7*STATE["rolling_acc_skill"] + 0.3*got_it
    STATE["hints"] = STATE["hints"]+1 if want_hint else max(0, STATE["hints"]-1)
    meta = f"Mode={mode} | p_next={p:.2f} | turns={STATE['turns']}"
    return ans, meta

# ---------- Minimal MCQ bank (keep or expand) ----------
MCQ_BANK = {
    "Fractions": [
        {"Q":"What is 1/2 + 1/5?","A":"7/10","B":"6/7","C":"3/5","D":"9/10","Correct":"A","Expl":"(5+2)/10 = 7/10."},
        {"Q":"Simplify 4/8","A":"1/4","B":"1/2","C":"2/3","D":"3/4","Correct":"B","Expl":"÷4 → 1/2."},
    ],
    "Decimals": [
        {"Q":"0.8 × 0.2 = ?","A":"0.16","B":"0.12","C":"0.18","D":"0.20","Correct":"A","Expl":"8×2=16; two decimals → 0.16."},
    ],
}

topic_choices = sorted(MCQ_BANK.keys()) if MCQ_BANK else []
default_topic = "Fractions" if "Fractions" in topic_choices else (topic_choices[0] if topic_choices else None)
print("[startup] topics found:", topic_choices)

P_SESS = {"topic": None, "pool": [], "idx": 0, "attempts": 0, "corrects": 0}

def start_practice(topic):
    if topic not in MCQ_BANK:
        return "Topic not found.", "", ""
    P_SESS.update({"topic": topic, "pool": random.sample(MCQ_BANK[topic], len(MCQ_BANK[topic])), "idx": 0,
                   "attempts": 0, "corrects": 0})
    q = P_SESS["pool"][0]
    text = f"{q['Q']}\nA){q['A']}\nB){q['B']}\nC){q['C']}\nD){q['D']}"
    status = f"Need {TOPIC_TARGET_CORRECT} correct to complete '{topic}'."
    return f"Started practice: {topic}", text, status

def check_answer(choice):
    if not P_SESS["pool"]:
        return "No quiz loaded.", "", ""
    q = P_SESS["pool"][P_SESS["idx"]]
    correct = q["Correct"].upper().strip()
    P_SESS["attempts"] += 1
    is_right = (choice and choice.upper().strip() == correct)
    if is_right:
        teacher = f"Correct because: {q['Expl']}"
        P_SESS["corrects"] += 1
        if P_SESS["corrects"] >= TOPIC_TARGET_CORRECT:
            return f"✅ Correct ({correct}). {teacher}", "", f"🎉 Topic '{P_SESS['topic']}' complete!"
        P_SESS["idx"] = (P_SESS["idx"] + 1) % len(P_SESS["pool"])
        nq = P_SESS["pool"][P_SESS["idx"]]
        text = f"{nq['Q']}\nA){nq['A']}\nB){nq['B']}\nC){nq['C']}\nD){nq['D']}"
        status = f"Score: {P_SESS['corrects']}/{P_SESS['attempts']}  •  Need {TOPIC_TARGET_CORRECT} to finish"
        return f"✅ Correct ({correct}). {teacher}", text, status
    else:
        hint = f"Hint: think about {q['Expl']}"
        text = f"{q['Q']}\nA){q['A']}\nB){q['B']}\nC){q['C']}\nD){q['D']}"
        status = f"Score: {P_SESS['corrects']}/{P_SESS['attempts']}  •  Need {TOPIC_TARGET_CORRECT} to finish"
        return f"❌ Incorrect. Correct is {correct}. {hint}", text, status

with gr.Blocks() as demo:
    gr.Markdown("## 📘 Personalized Tutor + Practice (Groq) — safe import")
    with gr.Tab("Tutor"):
        q = gr.Textbox(label="Ask a question")
        want_hint = gr.Checkbox(label="Hint only")
        ans = gr.Textbox(label="Tutor answer")
        dbg = gr.Textbox(label="Debug/State")
        gr.Button("Send").click(tutor, [q, want_hint], [ans, dbg])
    with gr.Tab("Practice"):
        if default_topic:
            topic = gr.Dropdown(choices=topic_choices, value=default_topic, label="Topic")
        else:
            topic = gr.Dropdown(choices=[], label="Topic", allow_custom_value=True)
        out1 = gr.Textbox(label="Status/Feedback")
        out2 = gr.Textbox(label="Question")
        out3 = gr.Textbox(label="Progress")
        gr.Button("Start Practice").click(start_practice, [topic], [out1, out2, out3])
        choice = gr.Dropdown(choices=["A","B","C","D"], label="Your Answer")
        gr.Button("Check Answer").click(check_answer, [choice], [out1, out2, out3])

