"""
Vidya Setu — Tutor (Groq) + Adaptive Multi-Topic Exam
Now with Student Progress Dataset (CSV by default, Firestore optional)
"""

import os, time, random, math, requests, csv, datetime, uuid
from typing import List, Tuple, Dict
from fastapi import FastAPI
import gradio as gr
import sympy as sp

# ---------------- Config ----------------
TOPIC_TARGET_CORRECT = int(os.getenv("TOPIC_TARGET_CORRECT", "5"))
DIFF_LEVELS = ["Easy", "Medium", "Hard"]

# Logging config
LOG_MODE = os.getenv("LOG_MODE", "csv").lower()            # "csv" or "firestore"
LOG_PATH = os.getenv("LOG_PATH", "progress.csv")           # CSV path (e.g., "/data/progress.csv" on Railway PV)
FS_COLLECTION = os.getenv("FS_COLLECTION", "attempts")     # Firestore collection name
USE_FIRESTORE = os.getenv("USE_FIRESTORE", "0") == "1"

# Try Firestore import (optional dependency)
try:
    from google.cloud import firestore  # type: ignore
except Exception:
    firestore = None

# If you deploy behind a subpath, set this as an ENV on the platform:
#   GRADIO_ROOT_PATH=/tutor
# Do NOT hardcode it here.

# ------------- Tutor (Groq) -------------
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_FALLBACKS = [
    os.getenv("GROQ_MODEL", "").strip() or "llama-3.3-70b-versatile",
    "gemma2-9b-it",
    "llama-3.1-8b-instant",
]
_LAST_WORKING_MODEL = None

def explain_with_groq(prompt: str) -> str:
    """Return a string; never raise so the UI still loads if key is wrong."""
    global _LAST_WORKING_MODEL
    key = (os.getenv("GROQ_API_KEY") or "").strip().strip('"').strip("'")
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
            if r.status_code in (401, 403):  # bad key
                return f"❌ Groq {r.status_code} {txt}"
            if r.status_code in (404, 422, 429):  # try another model
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

STATE_TUTOR = {"turns":0, "rolling_acc_user":0.7, "rolling_acc_skill":0.7, "hints":0}

def predict_next_prob():
    return 0.5*STATE_TUTOR["rolling_acc_user"] + 0.5*STATE_TUTOR["rolling_acc_skill"]

def adapt_mode(p):
    if p < 0.4: return "scaffold"
    if p < 0.7: return "normal"
    return "challenge"

def tutor(question, want_hint):
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

    STATE_TUTOR["turns"] += 1
    got_it = 0 if want_hint else 1
    STATE_TUTOR["rolling_acc_user"]  = 0.7*STATE_TUTOR["rolling_acc_user"]  + 0.3*got_it
    STATE_TUTOR["rolling_acc_skill"] = 0.7*STATE_TUTOR["rolling_acc_skill"] + 0.3*got_it
    STATE_TUTOR["hints"] = STATE_TUTOR["hints"]+1 if want_hint else max(0, STATE_TUTOR["hints"]-1)
    meta = f"Mode={mode} | p_next={p:.2f} | turns={STATE_TUTOR['turns']}"
    return ans, meta

# ------- Auto-generated MCQ generators -------
def _choices_with_distractors(correct, spread=5, count=4):
    opts = {str(correct)}
    try:
        base = float(correct)
        while len(opts) < count:
            delta = random.choice([-3, -2, -1, 1, 2, 3]) * (spread if isinstance(spread, (int, float)) else 1)
            opts.add(str(round(base + delta, 2)))
    except Exception:
        pool = ["Na","K","He","H","Ag","Fe","Cu","Zn","Pb","Sn","Ne","Ar","C","O"]
        while len(opts) < count:
            opts.add(random.choice(pool))
    out = list(opts); random.shuffle(out)
    return out

def q_fractions(diff: str) -> Tuple[str, str, List[str]]:
    rng = {"Easy": 9, "Medium": 12, "Hard": 20}[diff]
    a, b, c, d = random.randint(1, rng//2), random.randint(2, rng), random.randint(1, rng//2), random.randint(2, rng)
    op = random.choice(["+", "-", "×"])
    if op == "+": num, den = a*d + c*b, b*d; q = f"Compute {a}/{b} + {c}/{d} (simplify)."
    elif op == "-": num, den = a*d - c*b, b*d; q = f"Compute {a}/{b} - {c}/{d} (simplify)."
    else: num, den = a*c, b*d; q = f"Compute {a}/{b} × {c}/{d} (simplify)."
    g = math.gcd(abs(num), den); num //= g; den //= g
    ans = f"{num}/{den}" if den != 1 else f"{num}"
    opts = {ans}
    if den != 1:
        opts.add(f"{num+1}/{den}"); opts.add(f"{max(1,num)}/{max(1,den-1)}"); opts.add(f"{num+2}/{den}")
    else:
        opts |= {str(num+1), str(num+2), str(num-1)}
    choices = list(opts); random.shuffle(choices)
    return q, ans, choices

def q_decimals(diff: str):
    limit = {"Easy": 50, "Medium": 100, "Hard": 200}[diff]
    x = round(random.uniform(1, limit), 1)
    y = round(random.uniform(1, max(5, limit/5)), 1)
    op = random.choice(["+", "−", "×"])
    if op == "+": ans, q = round(x + y, 2), f"Compute {x} + {y}."
    elif op == "−": ans, q = round(x - y, 2), f"Compute {x} − {y}."
    else: ans, q = round(x * y, 2), f"Compute {x} × {y}."
    return q, str(ans), _choices_with_distractors(ans, spread=max(1, int(limit*0.02)))

def q_percentages(diff: str):
    base = random.randint(40, {"Easy": 300, "Medium": 600, "Hard": 1200}[diff])
    pct  = random.choice([5,8,10,12,15,18,20,25,30,40,50])
    kind = random.choice(["of","increase","decrease"])
    if kind == "of": ans = round(base * pct / 100, 2); q = f"Find {pct}% of {base}."
    elif kind == "increase": ans = round(base * (1 + pct/100), 2); q = f"{base} increased by {pct}% equals?"
    else: ans = round(base * (1 - pct/100), 2); q = f"{base} decreased by {pct}% equals?"
    return q, str(ans), _choices_with_distractors(ans, spread=max(2, int(base*0.05)))

def q_algebra(diff: str):
    rng = {"Easy": 12, "Medium": 25, "Hard": 40}[diff]
    a = random.randint(2, rng); x = random.randint(1, rng); b = random.randint(-rng//2, rng//2)
    c = a*x + b
    q = f"Solve for x: {a}x + ({b}) = {c}"
    return q, str(x), _choices_with_distractors(x, spread=max(3, rng//10))

def q_geometry(diff: str):
    kind = random.choice(["area_rect","perim_rect","area_circle","tri_area"])
    scale = {"Easy": 15, "Medium": 25, "Hard": 40}[diff]
    if kind == "area_rect":
        l,w = random.randint(3, scale), random.randint(3, scale); ans = l*w
        q = f"Area of rectangle (l={l} cm, w={w} cm)?"
    elif kind == "perim_rect":
        l,w = random.randint(3, scale), random.randint(3, scale); ans = 2*(l+w)
        q = f"Perimeter of rectangle (l={l} cm, w={w} cm)?"
    elif kind == "area_circle":
        r = random.randint(2, max(5, scale//2)); ans = round(3.14*r*r, 2)
        q = f"Area of circle (r={r} cm). Use π≈3.14."
    else:
        b,h = random.randint(4, scale), random.randint(4, scale); ans = 0.5*b*h
        q = f"Area of triangle (base={b} cm, height={h} cm)?"
    return q, str(ans), _choices_with_distractors(ans, spread=5)

def q_ratio(diff: str):
    limit = {"Easy": 80, "Medium": 200, "Hard": 400}[diff]
    a,b = random.randint(1,9), random.randint(1,9); total = random.randint(40, limit)
    part = random.choice(["first","second"]); s = a+b
    ans = (a if part=="first" else b) * total / s
    q = f"Divide {total} in the ratio {a}:{b}. What is the {part} part?"
    return q, str(int(ans)), _choices_with_distractors(int(ans), spread=5)

def q_hcflcm(diff: str):
    x,y = random.randint(6, 80 if diff!="Easy" else 40), random.randint(6, 80 if diff!="Easy" else 40)
    kind = random.choice(["HCF","LCM"]); g = math.gcd(x,y)
    ans = g if kind=="HCF" else x*y//g
    q = f"Find {kind} of {x} and {y}."
    return q, str(ans), _choices_with_distractors(ans, spread=5)

def q_sit(diff: str):
    p = random.randint(1000, {"Easy": 8000, "Medium": 15000, "Hard": 30000}[diff])
    r = random.choice([4,5,6,7,8,10,12,15]); t = random.choice([1,2,3,4,5])
    si = p*r*t/100
    q = f"Simple Interest on ₹{p} at {r}% p.a. for {t} years?"
    return q, str(int(si)), _choices_with_distractors(int(si), spread=100)

def q_sdt(diff: str):
    s = random.randint(20, {"Easy": 80, "Medium": 120, "Hard": 180}[diff])
    t = random.randint(1, 6); d = s*t
    kind = random.choice(["distance","time","speed"])
    if kind=="distance": q, ans = f"Speed {s} km/h for {t} h. Find distance (km).", d
    elif kind=="time":   q, ans = f"Distance {d} km at {s} km/h. Find time (h).", round(d/s,2)
    else:                q, ans = f"Distance {d} km in {t} h. Find speed (km/h).", s
    return q, str(ans), _choices_with_distractors(ans, spread=5)

CHEM = [
    ("Hydrogen","H",1), ("Helium","He",2), ("Lithium","Li",3), ("Beryllium","Be",4),
    ("Boron","B",5), ("Carbon","C",6), ("Nitrogen","N",7), ("Oxygen","O",8),
    ("Fluorine","F",9), ("Neon","Ne",10), ("Sodium","Na",11), ("Magnesium","Mg",12),
    ("Aluminium","Al",13), ("Silicon","Si",14), ("Phosphorus","P",15), ("Sulfur","S",16),
    ("Chlorine","Cl",17), ("Argon","Ar",18), ("Potassium","K",19), ("Calcium","Ca",20),
    ("Iron","Fe",26), ("Copper","Cu",29), ("Zinc","Zn",30), ("Silver","Ag",47), ("Gold","Au",79)
]
def q_chem(diff: str):
    name, sym, num = random.choice(CHEM)
    if random.choice([True, False]):
        q, ans = f"Symbol of **{name}**?", sym
        choices = _choices_with_distractors(ans, spread=2)
        if len(choices) < 4:  # ensure 4 options
            pool = [s for _, s, _ in CHEM]
            while len(choices) < 4:
                choices.append(random.choice(pool))
        random.shuffle(choices)
        return q, ans, choices
    else:
        q, ans = f"Atomic number of **{name}**?", str(num)
        return q, ans, _choices_with_distractors(ans, spread=2)

TOPICS: Dict[str, callable] = {
    "Fractions": q_fractions,
    "Decimals": q_decimals,
    "Percentages": q_percentages,
    "Algebra": q_algebra,
    "Geometry": q_geometry,
    "Ratio & Proportion": q_ratio,
    "HCF / LCM": q_hcflcm,
    "Simple Interest": q_sit,
    "Speed–Time–Distance": q_sdt,
    "Chemistry: Symbols & Atomic Numbers": q_chem,
}

# --------- AI helpers for the Exam tab ---------
def _has_groq():
    key = (os.getenv("GROQ_API_KEY") or "").strip()
    return key.startswith("gsk_")

def llm_explain_for_exam(topic, question_text, selected, correct):
    """Short, targeted explanation for a wrong answer."""
    if not _has_groq():
        return ""
    prompt = (
        "You are a concise teacher. First explain briefly why the selected option is wrong, "
        "then show the correct method in 2–3 short steps. Keep it under 120 words.\n\n"
        f"Topic: {topic}\nQuestion: {question_text}\n"
        f"Student chose: {selected}\nCorrect answer: {correct}"
    )
    out = explain_with_groq(prompt)
    return "\n\n**Why:** " + out if out else ""

def llm_followup(topic, question_text, correct):
    """Optional: a slightly harder follow-up MCQ (A–D) from same concept."""
    if not _has_groq():
        return ""
    prompt = (
        "Create ONE slightly harder follow-up MCQ (A–D) on the same concept. "
        "Return exactly:\nQuestion\nA) ...\nB) ...\nC) ...\nD) ...\nAnswer: X\n\n"
        f"Topic: {topic}\nSeed question: {question_text}\nSeed answer: {correct}"
    )
    out = explain_with_groq(prompt)
    return "\n\n**Try this:**\n" + out if out else ""

# --------- Logging (CSV / Firestore) ----------
def _csv_write_header_if_needed(path: str, fields: list[str]):
    newfile = not os.path.exists(path)
    if newfile:
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

def _log_csv(row: dict):
    fields = ["ts","student","session","topic_index","topic","difficulty","question",
              "answer","selected","is_correct","topic_correct","topic_attempts","overall_correct"]
    _csv_write_header_if_needed(LOG_PATH, fields)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writerow(row)

_FS_CLIENT = None
def _get_fs_client():
    global _FS_CLIENT
    if _FS_CLIENT is not None:
        return _FS_CLIENT
    if not (USE_FIRESTORE and firestore):
        return None
    # project can be auto-detected on Cloud Run; allow override via env
    try:
        project = os.getenv("FIRESTORE_PROJECT") or None
        _FS_CLIENT = firestore.Client(project=project)
        return _FS_CLIENT
    except Exception:
        return None

def _log_firestore(row: dict):
    client = _get_fs_client()
    if not client:
        return False
    try:
        client.collection(FS_COLLECTION).add(row)
        return True
    except Exception:
        return False

def log_attempt(row: dict):
    """Write to Firestore if enabled & available, else CSV."""
    ok = False
    if USE_FIRESTORE and firestore:
        ok = _log_firestore(row)
    if not ok:
        _log_csv(row)

# ------------- Build UI -------------
with gr.Blocks(title="Vidya Setu — Tutor + Adaptive Exam") as demo:
    gr.Markdown("# Vidya Setu — Personalized Tutor & Adaptive Exam")

    # ----- Tab 1: Tutor -----
    with gr.Tab("Tutor (Groq)"):
        q = gr.Textbox(label="Ask a question")
        want_hint = gr.Checkbox(label="Hint only")
        ans = gr.Textbox(label="Tutor answer")
        dbg = gr.Textbox(label="Debug/State")
        gr.Button("Send").click(tutor, [q, want_hint], [ans, dbg])

    # ----- Tab 2: Adaptive Exam -----
    with gr.Tab("Adaptive Exam"):
        gr.Markdown(
            f"**How it works:** Each topic continues until you get **{TOPIC_TARGET_CORRECT} correct**. "
            "Difficulty adapts up/down. When a topic reaches the target, we move to the next. Final report at the end."
        )

        student_id = gr.Textbox(label="Student ID (for progress tracking)", placeholder="e.g., 2025A001")
        topic_select = gr.CheckboxGroup(
            choices=list(TOPICS.keys()),
            value=["Fractions", "Decimals", "Percentages"],
            label="Select topics (in order)",
        )
        start_btn = gr.Button("Start Exam", variant="primary")

        status = gr.Markdown()
        question_md = gr.Markdown()
        options = gr.Radio(choices=[], label="Choose your answer")
        check_btn = gr.Button("Check")
        feedback = gr.Markdown()
        next_btn = gr.Button("Next ▶")

        with gr.Row():
            curr_topic = gr.Markdown()
            progress = gr.Markdown()
            score_box = gr.Markdown()

        report = gr.Markdown()
        # Optional: quick link to download CSV (works in most hosts)
        csv_path_show = gr.Textbox(value=LOG_PATH, label="CSV path", interactive=False)
        download_btn = gr.Button("Refresh CSV Path")

        # ✅ one shared state for this tab
        exam_state = gr.State({})

        # ---- handlers ----
        def gen_question(topic: str, diff_idx: int):
            diff = DIFF_LEVELS[diff_idx]
            q, ans, choices = TOPICS[topic](diff)
            return q, ans, choices, diff

        def adapt(up: bool, idx: int) -> int:
            return min(2, idx + 1) if up else max(0, idx - 1)

        def start_exam(student, topics):
            if not topics:
                return ("Please select at least one topic.", "", gr.update(choices=[], value=None),
                        "", "", "", {}, "", LOG_PATH)
            st = {
                "student": (student or "anon").strip(),
                "session": str(uuid.uuid4())[:8],
                "topics": topics,
                "topic_idx": 0,
                "diff_idx": 0,                # start Easy
                "correct_in_topic": 0,
                "attempts_in_topic": 0,
                "results": {t: {"correct": 0, "attempts": 0} for t in topics},
                "answer": None,
                "score_total": 0,
                "last_feedback": "",
                "last_question": "",
            }
            topic = st["topics"][0]
            q, ans, choices, diff = gen_question(topic, st["diff_idx"])
            st["answer"] = ans
            st["last_question"] = q
            status = f"**Exam started** for **{st['student']}** | Session: `{st['session']}` | Topics: {', '.join(topics)}"
            curr   = f"**Topic:** {topic}  |  **Difficulty:** {diff}"
            prog   = f"Correct in topic: **0 / {TOPIC_TARGET_CORRECT}**"
            score  = f"**Overall correct:** 0"
            return status, q, gr.update(choices=choices, value=None), curr, prog, score, st, "", LOG_PATH

        def check_answer(choice, st):
            if not st or st.get("answer") is None:
                return "Click **Start Exam** first.", st, "", LOG_PATH
            topic = st["topics"][st["topic_idx"]]
            st["attempts_in_topic"] += 1
            st["results"][topic]["attempts"] += 1

            is_correct = (choice is not None) and (str(choice).strip() == str(st["answer"]).strip())

            # --- log attempt ---
            row = {
                "ts": datetime.datetime.utcnow().isoformat(),
                "student": st.get("student","anon"),
                "session": st.get("session",""),
                "topic_index": st["topic_idx"],
                "topic": topic,
                "difficulty": DIFF_LEVELS[st["diff_idx"]],
                "question": st.get("last_question",""),
                "answer": st["answer"],
                "selected": (choice or ""),
                "is_correct": int(is_correct),
                "topic_correct": st["correct_in_topic"] + (1 if is_correct else 0),
                "topic_attempts": st["attempts_in_topic"],
                "overall_correct": st["score_total"] + (1 if is_correct else 0),
            }
            try:
                log_attempt(row)
            except Exception:
                pass  # never break the UI if logging fails

            # --- adapt + messaging ---
            if is_correct:
                st["correct_in_topic"] += 1
                st["results"][topic]["correct"] += 1
                st["score_total"] += 1
                follow = llm_followup(topic, st.get("last_question",""), st["answer"])
                msg = "✅ Correct!" + (f"{follow}" if follow else "")
                st["diff_idx"] = adapt(True, st["diff_idx"])
            else:
                ai_note = llm_explain_for_exam(topic, st.get("last_question",""), choice, st["answer"])
                msg = f"❌ Wrong. Correct answer: **{st['answer']}**" + (f"{ai_note}" if ai_note else "")
                st["diff_idx"] = adapt(False, st["diff_idx"])

            st["last_feedback"] = msg
            return msg, st, "", LOG_PATH

        def next_step(st):
            if not st or "topics" not in st:
                return ("Click **Start Exam** first.", "", gr.update(choices=[], value=None),
                        "", "", "", st, "", LOG_PATH)
            topics = st["topics"]
            topic  = topics[st["topic_idx"]]

            # Topic finished?
            if st["correct_in_topic"] >= TOPIC_TARGET_CORRECT:
                st["topic_idx"] += 1
                st["diff_idx"] = 0
                st["correct_in_topic"] = 0
                st["attempts_in_topic"] = 0

                if st["topic_idx"] >= len(topics):
                    # Final report
                    lines = [f"### 📊 Final Report — overall correct: **{st['score_total']}**  |  Student: **{st.get('student','anon')}**  |  Session: `{st.get('session','')}`"]
                    weak = []
                    for t, v in st["results"].items():
                        att = max(1, v["attempts"])
                        acc = round(100 * v["correct"] / att)
                        lines.append(f"- **{t}** — {v['correct']} / {v['attempts']}  (**{acc}%**)")
                        if acc < 80: weak.append((t, acc))
                    if weak:
                        lines.append("\n**Focus on:** " + ", ".join([f"{t} ({a}%)" for t,a in weak]))
                    if os.getenv("GROQ_API_KEY"):
                        lines.append("\n_Status: GROQ key present — AI hints available._")
                    report_md = "\n".join(lines)
                    return ("🎉 Exam finished!", "", gr.update(choices=[], value=None),
                            "", "", f"**Overall correct:** {st['score_total']}", st, report_md, LOG_PATH)

                # Move to first question of next topic
                topic = topics[st["topic_idx"]]
                q, ans, choices, diff = gen_question(topic, st["diff_idx"])
                st["answer"] = ans
                st["last_question"] = q
                curr = f"**Next topic:** {topic}  |  **Difficulty:** {diff}"
                prog = f"Correct in topic: **0 / {TOPIC_TARGET_CORRECT}**"
                score = f"**Overall correct:** {st['score_total']}"
                return ("➡️ Topic complete. Moving on.", q, gr.update(choices=choices, value=None),
                        curr, prog, score, st, "", LOG_PATH)

            # Still in same topic → new question
            q, ans, choices, diff = gen_question(topic, st["diff_idx"])
            st["answer"] = ans
            st["last_question"] = q
            curr = f"**Topic:** {topic}  |  **Difficulty:** {diff}"
            prog = f"Correct in topic: **{st['correct_in_topic']} / {TOPIC_TARGET_CORRECT}**"
            score = f"**Overall correct:** {st['score_total']}"
            return (st.get("last_feedback",""), q, gr.update(choices=choices, value=None),
                    curr, prog, score, st, "", LOG_PATH)

        # wire up (use the SAME state everywhere)
        start_btn.click(start_exam, [student_id, topic_select],
                        [status, question_md, options, curr_topic, progress, score_box, exam_state, report, csv_path_show])
        check_btn.click(check_answer, [options, exam_state],
                        [feedback, exam_state, report, csv_path_show])
        next_btn.click(next_step, [exam_state],
                       [feedback, question_md, options, curr_topic, progress, score_box, exam_state, report, csv_path_show])

        download_btn.click(lambda: LOG_PATH, [], [csv_path_show])

# ------------- FastAPI + mount -------------
app = FastAPI()

@app.get("/healthz")
def healthz():
    return {"ok": True}

app = gr.mount_gradio_app(app, demo, path="/")
