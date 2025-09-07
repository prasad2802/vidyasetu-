import os, time, random, requests
os.environ["GRADIO_ROOT_PATH"] = "/tutor"
import gradio as gr
import sympy as sp

# =========================
# Settings
# =========================
TOPIC_TARGET_CORRECT = 5  # need 5 correct answers to complete a topic
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# ==== Groq model fallback order (edit to your taste) ====
GROQ_FALLBACKS = [
    os.getenv("GROQ_MODEL", "").strip() or "llama-3.3-70b-versatile",
    "gemma2-9b-it",
    "llama-3.1-8b-instant",
    "deepseek-r1-distill-llama-70b",
    "groq.compound-mini",
    "allam-2-7b",
]

# Persist the last working model during the session
_LAST_WORKING_MODEL = None

# =========================
# Global state
# =========================
STATE = {
    "turns": 0,
    "seq_index": 0,
    "prev_correct_user": 0,
    "prev_correct_skill": 0,
    "rolling_acc_user": 0.7,
    "rolling_acc_skill": 0.7,
    "delta_t": 0,
    "hints": 0,
}

# =========================
# Groq caller (with automatic fallback)
# =========================
def explain_with_groq(prompt: str) -> str:
    """Try a list of Groq models until one succeeds; return clear errors otherwise."""
    global _LAST_WORKING_MODEL

    # Read and sanitize the key from env
    key = (os.getenv("GROQ_API_KEY") or "").strip().strip('"').strip("'")
    if not key.startswith("gsk_"):
        return "❌ GROQ_API_KEY missing/invalid on server."

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    # Build the try list: remember last success → try it first
    try_order = []
    if _LAST_WORKING_MODEL:
        try_order.append(_LAST_WORKING_MODEL)
    for m in GROQ_FALLBACKS:
        if m and m not in try_order:
            try_order.append(m)

    last_err = None
    for model in try_order:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        try:
            r = requests.post(GROQ_URL, headers=headers, json=body, timeout=60)
            if r.status_code == 200:
                _LAST_WORKING_MODEL = model
                data = r.json()
                content = data["choices"][0]["message"]["content"].strip()
                return f"(model: {model})\n{content}"

            # Non-200 handling and fallbacks
            txt = r.text[:350]
            if r.status_code in (401, 403):
                return f"❌ Groq {r.status_code} for {model}: {txt}"
            if r.status_code in (404, 422):
                last_err = f"ℹ️ Skipped {model}: {txt}"
                continue
            if r.status_code == 429:
                last_err = f"⏳ Rate limit on {model}: trying another… Details: {txt}"
                continue

            last_err = f"❌ HTTP {r.status_code} on {model}: {txt}"

        except Exception as e:
            last_err = f"❌ Request failed on {model}: {type(e).__name__} → {e}"

    return last_err or "❌ All Groq model attempts failed."

# =========================
# Math Guard
# =========================
def solve_math(expr: str):
    try:
        expr = expr.replace("=", "")
        res = sp.simplify(expr)
        return str(res)
    except Exception:
        return None

# =========================
# Personalization (simple)
# =========================
def predict_next_prob():
    return 0.5 * STATE["rolling_acc_user"] + 0.5 * STATE["rolling_acc_skill"]

def adapt_mode(p):
    if p < 0.4:
        return "scaffold"
    elif p < 0.7:
        return "normal"
    else:
        return "challenge"

# =========================
# Tutor
# =========================
def tutor(question, want_hint):
    t0 = time.time()
    p = predict_next_prob()
    mode = "scaffold" if want_hint else adapt_mode(p)
    style = {
        "scaffold": "Explain step-by-step in simple words. Use a short numbered list. End with one tiny practice.",
        "normal": "Explain clearly in 2–3 steps with a small example.",
        "challenge": "Give a rigorous explanation and add a harder follow-up question.",
    }[mode]

    looks_math = any(op in question for op in "+-*/=")
    computed = solve_math(question) if looks_math else None

    if computed is not None:
        prompt = f"You are a math teacher. {style}\nExplain why the result of '{question}' is {computed}."
        ans = explain_with_groq(prompt)
        ans = f"✅ Answer: {computed}\n{ans}"
    else:
        prompt = f"You are a patient school tutor. {style}\nQuestion: {question}\nAnswer:"
        ans = explain_with_groq(prompt)

    # update state
    STATE["turns"] += 1
    STATE["seq_index"] += 1
    got_it = 0 if want_hint else 1
    STATE["prev_correct_user"] = got_it
    STATE["prev_correct_skill"] = got_it
    STATE["rolling_acc_user"] = 0.7 * STATE["rolling_acc_user"] + 0.3 * got_it
    STATE["rolling_acc_skill"] = 0.7 * STATE["rolling_acc_skill"] + 0.3 * got_it
    STATE["delta_t"] = time.time() - t0
    STATE["hints"] = STATE["hints"] + 1 if want_hint else max(0, STATE["hints"] - 1)

    active_model = _LAST_WORKING_MODEL or GROQ_FALLBACKS[0]
    meta = f"Mode={mode} | p_next={p:.2f} | turns={STATE['turns']} | hints={STATE['hints']} | model={active_model}"
    return ans, meta

# =========================
# MCQ Bank (Math + Chemistry)
# =========================
MCQ_BANK = {
    # (your MCQ_BANK exactly as you had it)
    # -------------- keep the rest unchanged --------------
}

# Practice session state
P_SESS = {"topic": None, "pool": [], "idx": 0, "attempts": 0, "corrects": 0}

# =========================
# Practice logic
# =========================
def start_practice(topic):
    if topic not in MCQ_BANK:
        return "Topic not found.", "", ""
    P_SESS["topic"] = topic
    P_SESS["pool"] = random.sample(MCQ_BANK[topic], len(MCQ_BANK[topic]))
    P_SESS["idx"] = 0
    P_SESS["attempts"] = 0
    P_SESS["corrects"] = 0
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
        prompt = (
            f"Explain briefly why the correct option is {correct}.\n"
            f"Question: {q['Q']}\n"
            f"Options: A){q['A']}  B){q['B']}  C){q['C']}  D){q['D']}\n"
            f"Keep it to 1–2 short lines for grade 6."
        )
        teacher = explain_with_groq(prompt)

        P_SESS["corrects"] += 1

        if P_SESS["corrects"] >= TOPIC_TARGET_CORRECT:
            return (
                f"✅ Correct ({correct}). {teacher}",
                "",
                f"🎉 Topic '{P_SESS['topic']}' complete! Correct={P_SESS['corrects']}/{P_SESS['attempts']}",
            )

        P_SESS["idx"] += 1
        if P_SESS["idx"] >= len(P_SESS["pool"]):
            P_SESS["pool"] = random.sample(MCQ_BANK[P_SESS["topic"]], len(MCQ_BANK[P_SESS["topic"]]))
            P_SESS["idx"] = 0

        nq = P_SESS["pool"][P_SESS["idx"]]
        text = f"{nq['Q']}\nA){nq['A']}\nB){nq['B']}\nC){nq['C']}\nD){nq['D']}"
        status = f"Score: {P_SESS['corrects']}/{P_SESS['attempts']}  •  Need {TOPIC_TARGET_CORRECT} to finish"
        return f"✅ Correct ({correct}). {teacher}", text, status

    else:
        prompt = f"Give a short hint (<=2 lines) for this question without revealing the answer:\n{q['Q']}"
        hint = explain_with_groq(prompt)
        text = f"{q['Q']}\nA){q['A']}\nB){q['B']}\nC){q['C']}\nD){q['D']}"
        status = f"Score: {P_SESS['corrects']}/{P_SESS['attempts']}  •  Need {TOPIC_TARGET_CORRECT} to finish"
        return f"❌ Incorrect. Correct is {correct}. Hint: {hint}", text, status

# =========================
# UI
# =========================
with gr.Blocks() as demo:
    gr.Markdown("## 📘 Personalized Tutor + Practice (Groq) — 5-correct target / Math + Chemistry (with auto-fallback)")

    with gr.Tab("Tutor"):
        q = gr.Textbox(label="Ask a question")
        hint = gr.Checkbox(label="Hint only")
        ans = gr.Textbox(label="Tutor answer")
        dbg = gr.Textbox(label="Debug/State")
        gr.Button("Send").click(tutor, [q, hint], [ans, dbg])

    with gr.Tab("Practice"):
        topic = gr.Dropdown(choices=list(MCQ_BANK.keys()), value="Fractions", label="Topic")
        out1 = gr.Textbox(label="Status/Feedback")
        out2 = gr.Textbox(label="Question")
        out3 = gr.Textbox(label="Progress")
        start = gr.Button("Start Practice")
        start.click(start_practice, [topic], [out1, out2, out3])
        choice = gr.Dropdown(choices=["A", "B", "C", "D"], label="Your Answer")
        check = gr.Button("Check Answer")
        check.click(check_answer, [choice], [out1, out2, out3])

