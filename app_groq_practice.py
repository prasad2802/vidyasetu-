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
    os.getenv("GROQ_MODEL", "").strip() or "llama-3.3-70b-versatile",  # preferred (if you set GROQ_MODEL, it’s used 1st)
    "gemma2-9b-it",                   # reliable free model
    "llama-3.1-8b-instant",           # fast/light
    "deepseek-r1-distill-llama-70b",  # extra from your quota page
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

    key = (os.getenv("GROQ_API_KEY") or "").strip()
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
            "temperature": 0.2
        }
        try:
            r = requests.post(GROQ_URL, headers=headers, json=body, timeout=60)
            if r.status_code == 200:
                _LAST_WORKING_MODEL = model  # cache the good one for this session
                data = r.json()
                content = data["choices"][0]["message"]["content"].strip()
                return f"(model: {model})\n{content}"

            # Non-200 → decide whether to keep trying or stop
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
            continue

        except Exception as e:
            last_err = f"❌ Request failed on {model}: {type(e).__name__} → {e}"
            continue

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
    if p < 0.4: return "scaffold"
    elif p < 0.7: return "normal"
    else: return "challenge"

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
        "challenge": "Give a rigorous explanation and add a harder follow-up question."
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
    STATE["turns"] += 1; STATE["seq_index"] += 1
    got_it = 0 if want_hint else 1
    STATE["prev_correct_user"] = got_it; STATE["prev_correct_skill"] = got_it
    STATE["rolling_acc_user"] = 0.7*STATE["rolling_acc_user"] + 0.3*got_it
    STATE["rolling_acc_skill"] = 0.7*STATE["rolling_acc_skill"] + 0.3*got_it
    STATE["delta_t"] = time.time()-t0
    STATE["hints"] = STATE["hints"]+1 if want_hint else max(0, STATE["hints"]-1)

    active_model = _LAST_WORKING_MODEL or GROQ_FALLBACKS[0]
    meta = f"Mode={mode} | p_next={p:.2f} | turns={STATE['turns']} | hints={STATE['hints']} | model={active_model}"
    return ans, meta

# =========================
# MCQ Bank (Math + Chemistry)
# =========================
MCQ_BANK = {
    # --- Math ---
    "Fractions": [
        {"Q":"What is 1/2 + 1/5?","A":"7/10","B":"6/7","C":"3/5","D":"9/10","Correct":"A","Expl":"Common denominator 10 → (5+2)/10 = 7/10."},
        {"Q":"Simplify 4/8","A":"1/4","B":"1/2","C":"2/3","D":"3/4","Correct":"B","Expl":"Divide numerator & denominator by 4 → 1/2."},
        {"Q":"(2/3) × (3/5) = ?","A":"6/15","B":"2/5","C":"3/10","D":"5/6","Correct":"B","Expl":"(2×3)/(3×5) = 6/15 = 2/5."},
        {"Q":"Which is larger?","A":"3/4","B":"2/3","C":"5/8","D":"1/2","Correct":"A","Expl":"3/4=0.75; 2/3≈0.667; 5/8=0.625; 1/2=0.5."},
        {"Q":"What is 3/4 − 1/8?","A":"5/8","B":"2/3","C":"1/2","D":"3/8","Correct":"A","Expl":"LCM 8 → 6/8 − 1/8 = 5/8."},
        {"Q":"Simplify 9/12","A":"3/4","B":"2/3","C":"1/3","D":"3/5","Correct":"A","Expl":"Divide by 3 → 3/4."},
    ],
    "Decimals": [
        {"Q":"0.25 + 0.5 = ?","A":"0.55","B":"0.65","C":"0.75","D":"0.85","Correct":"C","Expl":"0.25 + 0.50 = 0.75."},
        {"Q":"0.8 × 0.2 = ?","A":"0.16","B":"0.12","C":"0.18","D":"0.20","Correct":"A","Expl":"8×2=16; two decimal places → 0.16."},
        {"Q":"Round 5.678 to 2 decimals","A":"5.68","B":"5.67","C":"5.70","D":"5.60","Correct":"A","Expl":"Third digit 8→ round 5.67 up to 5.68."},
        {"Q":"6.3 ÷ 0.9 = ?","A":"0.7","B":"7","C":"6.9","D":"0.14","Correct":"B","Expl":"Scale ×10: 63 ÷ 9 = 7."},
        {"Q":"Which is greater?","A":"0.25","B":"0.205","C":"0.2","D":"0.024","Correct":"A","Expl":"0.25 > 0.205 > 0.2 > 0.024."},
        {"Q":"0.4 × 0.06 = ?","A":"0.024","B":"0.24","C":"0.0024","D":"0.026","Correct":"A","Expl":"4×6=24; total 3 decimal places → 0.024."},
    ],
    "Percents": [
        {"Q":"10% of 250 = ?","A":"25","B":"2.5","C":"0.25","D":"2500","Correct":"A","Expl":"10% = 0.1; 0.1×250=25."},
        {"Q":"Increase 120 by 25%","A":"145","B":"150","C":"155","D":"125","Correct":"B","Expl":"120×1.25 = 150."},
        {"Q":"Convert 45% to decimal","A":"0.45","B":"4.5","C":"0.045","D":"45","Correct":"A","Expl":"45/100 = 0.45."},
        {"Q":"Decrease 80 by 15%","A":"72","B":"68","C":"70","D":"78","Correct":"B","Expl":"80×0.85 = 68."},
        {"Q":"What is 30% of 90?","A":"27","B":"30","C":"21","D":"33","Correct":"A","Expl":"0.30×90=27."},
        {"Q":"Which is larger?","A":"35%","B":"0.34","C":"3/10","D":"32%","Correct":"A","Expl":"35% = 0.35; others < 0.35."},
    ],
    "Geometry (Triangles)": [
        {"Q":"In a right triangle, a=3, b=4. Hypotenuse?","A":"5","B":"6","C":"7","D":"4.5","Correct":"A","Expl":"Pythagoras: √(3²+4²)=√25=5."},
        {"Q":"Sum of interior angles of a triangle?","A":"90°","B":"180°","C":"270°","D":"360°","Correct":"B","Expl":"Always 180°."},
        {"Q":"An isosceles triangle has…","A":"all equal sides","B":"two equal sides","C":"no equal sides","D":"right angle","Correct":"B","Expl":"Isosceles ⇒ exactly two equal sides."},
        {"Q":"Area of right triangle (legs 6,8)?","A":"24","B":"28","C":"30","D":"48","Correct":"A","Expl":"Area = ½×base×height = ½×6×8 = 24."},
        {"Q":"If sides are 5,12,13, triangle is…","A":"acute","B":"right","C":"obtuse","D":"equilateral","Correct":"B","Expl":"5²+12²=25+144=169=13² → right."},
        {"Q":"Which can be triangle sides?","A":"2,3,6","B":"2,2,3","C":"1,1,3","D":"4,1,2","Correct":"B","Expl":"Triangle inequality holds only for 2,2,3."},
    ],
    "Algebra (Basics)": [
        {"Q":"Solve: x + 5 = 12","A":"x=5","B":"x=7","C":"x=12","D":"x=17","Correct":"B","Expl":"Subtract 5 both sides → x=7."},
        {"Q":"Solve: 3x = 21","A":"x=9","B":"x=7","C":"x=6","D":"x=63","Correct":"B","Expl":"Divide both sides by 3 → x=7."},
        {"Q":"Value of 2x+3 when x=4","A":"11","B":"10","C":"8","D":"15","Correct":"A","Expl":"2×4+3 = 8+3 = 11."},
        {"Q":"Simplify: 5x + 2x","A":"7","B":"7x","C":"x^7","D":"5x^2","Correct":"B","Expl":"Like terms add: 5x+2x=7x."},
        {"Q":"Solve: x−3=−1","A":"x=2","B":"x=−2","C":"x=1","D":"x=4","Correct":"A","Expl":"Add 3 to both sides → x=2."},
        {"Q":"If y=2x and x=5, y=?","A":"7","B":"10","C":"2","D":"25","Correct":"B","Expl":"y=2×5=10."},
    ],
    "Ratios & Proportion": [
        {"Q":"Ratio 6:9 simplified","A":"2:3","B":"3:2","C":"1:3","D":"3:1","Correct":"A","Expl":"Divide both by 3 → 2:3."},
        {"Q":"If 2/3 = x/9, x=?","A":"3","B":"4","C":"5","D":"6","Correct":"D","Expl":"Cross-multiply: 2×9=18; 18/3=6."},
        {"Q":"Proportion true?","A":"4/6 = 2/3","B":"5/8 = 3/5","C":"2/7 = 1/4","D":"3/9 = 2/3","Correct":"A","Expl":"4/6 simplifies to 2/3."},
        {"Q":"Divide ₹60 in ratio 2:3","A":"₹24 and ₹36","B":"₹20 and ₹40","C":"₹30 and ₹30","D":"₹10 and ₹50","Correct":"A","Expl":"Total parts 5; each part 12; 2×12=24, 3×12=36."},
        {"Q":"Scale a recipe 2×: 3 cups flour becomes…","A":"5 cups","B":"6 cups","C":"4 cups","D":"3 cups","Correct":"B","Expl":"Double of 3 is 6."},
        {"Q":"Which ratio equals 0.5?","A":"3:5","B":"1:3","C":"1:2","D":"2:1","Correct":"C","Expl":"1:2 = 0.5."},
    ],

    # --- Chemistry ---
    "Chemistry (Matter)": [
        {"Q":"Matter is anything that…","A":"has mass and occupies space","B":"is visible only","C":"has no volume","D":"has no mass","Correct":"A","Expl":"Definition of matter."},
        {"Q":"Which is NOT a state of matter?","A":"Solid","B":"Liquid","C":"Gas","D":"Light","Correct":"D","Expl":"Light is energy, not matter."},
        {"Q":"Particles in solids are…","A":"far apart","B":"free to flow","C":"closely packed","D":"ionized","Correct":"C","Expl":"Solids have tightly packed particles."},
        {"Q":"Change from liquid to gas is…","A":"Condensation","B":"Evaporation","C":"Freezing","D":"Sublimation","Correct":"B","Expl":"Liquid → gas = evaporation/boiling."},
        {"Q":"Sublimation is…","A":"solid→gas","B":"gas→solid","C":"liquid→solid","D":"liquid→gas","Correct":"A","Expl":"Direct solid to gas (e.g., dry ice)."},
        {"Q":"A mixture that looks uniform is…","A":"heterogeneous","B":"homogeneous","C":"colloidal only","D":"compound","Correct":"B","Expl":"Homogeneous looks uniform (e.g., salt water)."},
    ],
    "Chemistry (Elements & Symbols)": [
        {"Q":"Symbol for Hydrogen","A":"H","B":"He","C":"Hy","D":"Hg","Correct":"A","Expl":"Hydrogen = H."},
        {"Q":"Symbol for Sodium","A":"So","B":"Na","C":"Sn","D":"Sd","Correct":"B","Expl":"Sodium = Na (from ‘Natrium’)."},
        {"Q":"Symbol for Potassium","A":"P","B":"Pt","C":"K","D":"Po","Correct":"C","Expl":"Potassium = K (from ‘Kalium’)."},
        {"Q":"Symbol for Iron","A":"Ir","B":"Fe","C":"In","D":"I","Correct":"B","Expl":"Iron = Fe (from ‘Ferrum’)."},
        {"Q":"Symbol for Carbon","A":"C","B":"Ca","C":"Co","D":"Cr","Correct":"A","Expl":"Carbon = C."},
        {"Q":"Symbol for Oxygen","A":"O","B":"Ox","C":"Og","D":"Oy","Correct":"A","Expl":"Oxygen = O."},
    ],
    "Chemistry (Periodic Table Basics)": [
        {"Q":"Atomic number is…","A":"protons in nucleus","B":"neutrons in nucleus","C":"electrons in outer shell","D":"mass of atom","Correct":"A","Expl":"Atomic number = number of protons."},
        {"Q":"Element with atomic number 1","A":"Helium","B":"Hydrogen","C":"Lithium","D":"Oxygen","Correct":"B","Expl":"H has Z=1."},
        {"Q":"Group 1 elements are called…","A":"Halogens","B":"Noble gases","C":"Alkali metals","D":"Metalloids","Correct":"C","Expl":"Group 1 = alkali metals."},
        {"Q":"Noble gases are in…","A":"Group 17","B":"Group 18","C":"Period 2 only","D":"Transition block","Correct":"B","Expl":"Group 18 = noble gases (He, Ne, Ar…)."},
        {"Q":"Chlorine’s symbol","A":"Cl","B":"Ch","C":"Cr","D":"Cn","Correct":"A","Expl":"Chlorine = Cl."},
        {"Q":"Across a period, atomic size generally…","A":"increases","B":"decreases","C":"same","D":"random","Correct":"B","Expl":"Effective nuclear charge ↑ → size ↓."},
    ],
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

        # ✅ Finish topic when target met
        if P_SESS["corrects"] >= TOPIC_TARGET_CORRECT:
            return (
                f"✅ Correct ({correct}). {teacher}",
                "",
                f"🎉 Topic '{P_SESS['topic']}' complete! Correct={P_SESS['corrects']}/{P_SESS['attempts']}"
            )

        # Otherwise move to next question
        P_SESS["idx"] += 1
        if P_SESS["idx"] >= len(P_SESS["pool"]):
            # reshuffle more from the same topic (loop through the bank again)
            P_SESS["pool"] = random.sample(MCQ_BANK[P_SESS["topic"]], len(MCQ_BANK[P_SESS["topic"]]))
            P_SESS["idx"] = 0

        nq = P_SESS["pool"][P_SESS["idx"]]
        text = f"{nq['Q']}\nA){nq['A']}\nB){nq['B']}\nC){nq['C']}\nD){nq['D']}"
        status = f"Score: {P_SESS['corrects']}/{P_SESS['attempts']}  •  Need {TOPIC_TARGET_CORRECT} to finish"
        return f"✅ Correct ({correct}). {teacher}", text, status

    else:
        # Wrong → give a short hint; stay in same topic
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
        choice = gr.Dropdown(choices=["A","B","C","D"], label="Your Answer")
        check = gr.Button("Check Answer")
        check.click(check_answer, [choice], [out1, out2, out3])



