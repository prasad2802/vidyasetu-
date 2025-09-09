"""
Vidya Setu — Tutor (Groq) + Adaptive Multi-Topic Exam
Kid-friendly UI • AI explanations • CSV/Firestore logging • Final report download

This file is intentionally verbose with comments so new contributors (and you, months later!)
can understand how each piece works and where to extend it.
"""

import os, random, math, requests, csv, datetime, uuid
from typing import List, Tuple, Dict
from fastapi import FastAPI
import gradio as gr
import sympy as sp

# =========================
# Global Configuration
# =========================
# Target number of correct answers needed to complete each topic.
TOPIC_TARGET_CORRECT = int(os.getenv("TOPIC_TARGET_CORRECT", "5"))

# Difficulty levels are referred to by index (0..2) and name for display.
DIFF_LEVELS = ["Easy", "Medium", "Hard"]

# --- Logging: CSV by default; Firestore optional ---
LOG_MODE = os.getenv("LOG_MODE", "csv").lower()            # "csv" or "firestore" (CSV is fallback)
LOG_PATH = os.getenv("LOG_PATH", "progress.csv")           # Where attempts are appended
FS_COLLECTION = os.getenv("FS_COLLECTION", "attempts")     # Firestore collection name
USE_FIRESTORE = os.getenv("USE_FIRESTORE", "0") == "1"

# Directory to save final per-session Markdown reports.
REPORT_DIR = os.getenv("REPORT_DIR", "reports")

# Optional Firestore import (only required if using Firestore).
try:
    from google.cloud import firestore  # type: ignore
except Exception:
    firestore = None  # keeps the app working without that package

# =========================
# LLM (Groq) Tutor Helpers
# =========================
# We use Groq's OpenAI-compatible endpoint for explanations and follow-ups.
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_FALLBACKS = [
    os.getenv("GROQ_MODEL", "").strip() or "llama-3.3-70b-versatile",
    "gemma2-9b-it",
    "llama-3.1-8b-instant",
]
_LAST_WORKING_MODEL = None  # sticky cache of last good model to reduce retries


def explain_with_groq(prompt: str) -> str:
    """
    Call Groq's API with a conservative temperature to keep the responses short and stable.
    Never raise exceptions (UI should continue to work even without a valid API key).
    """
    global _LAST_WORKING_MODEL
    key = (os.getenv("GROQ_API_KEY") or "").strip().strip('"').strip("'")
    if not key.startswith("gsk_"):
        return "❌ GROQ_API_KEY missing/invalid on server."

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    # Try the last known working model first, then fall back to a small list.
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
                json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2},
                timeout=60,
            )
            if r.status_code == 200:
                _LAST_WORKING_MODEL = model
                return r.json()["choices"][0]["message"]["content"].strip()
            txt = r.text[:350]
            if r.status_code in (401, 403):  # bad key or permission
                return f"❌ Groq {r.status_code} {txt}"
            if r.status_code in (404, 422, 429):  # switch model
                last_err = f"ℹ️ {r.status_code} on {model}: {txt}"
                continue
            last_err = f"❌ HTTP {r.status_code} on {model}: {txt}"
        except Exception as e:
            last_err = f"❌ Request failed on {model}: {type(e).__name__}: {e}"
    return last_err or "❌ All Groq model attempts failed."


def solve_math(expr: str):
    """
    Try to compute a direct result using SymPy for basic math strings.
    Returns a simplified string or None if input is not a math expression.
    """
    try:
        expr = expr.replace("=", "")
        return str(sp.simplify(expr))
    except Exception:
        return None


# Lightweight, local state for the Tutor tab (not persisted).
STATE_TUTOR = {"turns": 0, "rolling_acc_user": 0.7, "rolling_acc_skill": 0.7, "hints": 0}


def predict_next_prob():
    """Heuristic 'probability of next success' from tutor usage."""
    return 0.5 * STATE_TUTOR["rolling_acc_user"] + 0.5 * STATE_TUTOR["rolling_acc_skill"]


def adapt_mode(p):
    """Map predicted success to an explanation style."""
    if p < 0.4:
        return "scaffold"
    if p < 0.7:
        return "normal"
    return "challenge"


def tutor(question, want_hint):
    """
    Main handler for the Tutor tab.
    - If the prompt looks mathematical, compute first with SymPy then ask the LLM to explain.
    - Otherwise, ask the LLM to answer in the requested style.
    """
    p = predict_next_prob()
    mode = "scaffold" if want_hint else adapt_mode(p)
    style = {
        "scaffold": "Explain step-by-step in simple words. Use a short list. End with one tiny practice.",
        "normal": "Explain clearly in 2–3 steps with a small example.",
        "challenge": "Give a rigorous explanation and add a harder follow-up question.",
    }[mode]

    looks_math = any(op in question for op in "+-*/=")
    computed = solve_math(question) if looks_math else None
    if computed is not None:
        prompt = f"You are a math teacher. {style}\nExplain why the result of '{question}' is {computed}."
        ans = f"✅ Answer: {computed}\n" + explain_with_groq(prompt)
    else:
        ans = explain_with_groq(f"You are a patient tutor. {style}\nQuestion: {question}\nAnswer:")

    # Update simple rolling accuracy to nudge the style over time.
    STATE_TUTOR["turns"] += 1
    got_it = 0 if want_hint else 1
    STATE_TUTOR["rolling_acc_user"] = 0.7 * STATE_TUTOR["rolling_acc_user"] + 0.3 * got_it
    STATE_TUTOR["rolling_acc_skill"] = 0.7 * STATE_TUTOR["rolling_acc_skill"] + 0.3 * got_it
    STATE_TUTOR["hints"] = STATE_TUTOR["hints"] + 1 if want_hint else max(0, STATE_TUTOR["hints"] - 1)
    meta = f"Mode={mode} | p_next={p:.2f} | turns={STATE_TUTOR['turns']}"
    return ans, meta


# =========================
# Question Generators (MCQ)
# =========================
# Pure-python item bank for core topics. Each function returns:
# (question_text, correct_answer, list_of_choices)

def _choices_with_distractors(correct, spread=5, count=4):
    """Build 3 distractors near the numeric answer (or use a symbol pool)."""
    opts = {str(correct)}
    try:
        base = float(correct)
        while len(opts) < count:
            delta = random.choice([-3, -2, -1, 1, 2, 3]) * (spread if isinstance(spread, (int, float)) else 1)
            opts.add(str(round(base + delta, 2)))
    except Exception:
        pool = ["Na", "K", "He", "H", "Ag", "Fe", "Cu", "Zn", "Pb", "Sn", "Ne", "Ar", "C", "O"]
        while len(opts) < count:
            opts.add(random.choice(pool))
    out = list(opts)
    random.shuffle(out)
    return out


def q_fractions(diff: str) -> Tuple[str, str, List[str]]:
    """Add/Subtract/Multiply fractions and simplify."""
    rng = {"Easy": 9, "Medium": 12, "Hard": 20}[diff]
    a, b, c, d = random.randint(1, rng // 2), random.randint(2, rng), random.randint(1, rng // 2), random.randint(2, rng)
    op = random.choice(["+", "-", "×"])
    if op == "+":
        num, den = a * d + c * b, b * d
        q = f"Compute {a}/{b} + {c}/{d} (simplify)."
    elif op == "-":
        num, den = a * d - c * b, b * d
        q = f"Compute {a}/{b} - {c}/{d} (simplify)."
    else:
        num, den = a * c, b * d
        q = f"Compute {a}/{b} × {c}/{d} (simplify)."
    g = math.gcd(abs(num), den)
    num //= g
    den //= g
    ans = f"{num}/{den}" if den != 1 else f"{num}"
    opts = {ans}
    if den != 1:
        opts.add(f"{num+1}/{den}")
        opts.add(f"{max(1, num)}/{max(1, den-1)}")
        opts.add(f"{num+2}/{den}")
    else:
        opts |= {str(num + 1), str(num + 2), str(num - 1)}
    choices = list(opts)
    random.shuffle(choices)
    return q, ans, choices


def q_decimals(diff: str):
    """Basic decimal arithmetic."""
    limit = {"Easy": 50, "Medium": 100, "Hard": 200}[diff]
    x = round(random.uniform(1, limit), 1)
    y = round(random.uniform(1, max(5, limit / 5)), 1)
    op = random.choice(["+", "−", "×"])
    if op == "+":
        ans, q = round(x + y, 2), f"Compute {x} + {y}."
    elif op == "−":
        ans, q = round(x - y, 2), f"Compute {x} − {y}."
    else:
        ans, q = round(x * y, 2), f"Compute {x} × {y}."
    return q, str(ans), _choices_with_distractors(ans, spread=max(1, int(limit * 0.02)))


def q_percentages(diff: str):
    """Percentage of / increase / decrease."""
    base = random.randint(40, {"Easy": 300, "Medium": 600, "Hard": 1200}[diff])
    pct = random.choice([5, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50])
    kind = random.choice(["of", "increase", "decrease"])
    if kind == "of":
        ans = round(base * pct / 100, 2)
        q = f"Find {pct}% of {base}."
    elif kind == "increase":
        ans = round(base * (1 + pct / 100), 2)
        q = f"{base} increased by {pct}% equals?"
    else:
        ans = round(base * (1 - pct / 100), 2)
        q = f"{base} decreased by {pct}% equals?"
    return q, str(ans), _choices_with_distractors(ans, spread=max(2, int(base * 0.05)))


def q_algebra(diff: str):
    """Solve ax + b = c."""
    rng = {"Easy": 12, "Medium": 25, "Hard": 40}[diff]
    a = random.randint(2, rng)
    x = random.randint(1, rng)
    b = random.randint(-rng // 2, rng // 2)
    c = a * x + b
    q = f"Solve for x: {a}x + ({b}) = {c}"
    return q, str(x), _choices_with_distractors(x, spread=max(3, rng // 10))


def q_geometry(diff: str):
    """Areas & perimeter with simple numbers."""
    kind = random.choice(["area_rect", "perim_rect", "area_circle", "tri_area"])
    scale = {"Easy": 15, "Medium": 25, "Hard": 40}[diff]
    if kind == "area_rect":
        l, w = random.randint(3, scale), random.randint(3, scale)
        ans = l * w
        q = f"Area of rectangle (l={l} cm, w={w} cm)?"
    elif kind == "perim_rect":
        l, w = random.randint(3, scale), random.randint(3, scale)
        ans = 2 * (l + w)
        q = f"Perimeter of rectangle (l={l} cm, w={w} cm)?"
    elif kind == "area_circle":
        r = random.randint(2, max(5, scale // 2))
        ans = round(3.14 * r * r, 2)
        q = f"Area of circle (r={r} cm). Use π≈3.14."
    else:
        b, h = random.randint(4, scale), random.randint(4, scale)
        ans = 0.5 * b * h
        q = f"Area of triangle (base={b} cm, height={h} cm)?"
    return q, str(ans), _choices_with_distractors(ans, spread=5)


def q_ratio(diff: str):
    """Simple ratio partition problems."""
    limit = {"Easy": 80, "Medium": 200, "Hard": 400}[diff]
    a, b = random.randint(1, 9), random.randint(1, 9)
    total = random.randint(40, limit)
    part = random.choice(["first", "second"])
    s = a + b
    ans = (a if part == "first" else b) * total / s
    q = f"Divide {total} in the ratio {a}:{b}. What is the {part} part?"
    return q, str(int(ans)), _choices_with_distractors(int(ans), spread=5)


def q_hcflcm(diff: str):
    """HCF/LCM of two integers."""
    x, y = random.randint(6, 80 if diff != "Easy" else 40), random.randint(6, 80 if diff != "Easy" else 40)
    kind = random.choice(["HCF", "LCM"])
    g = math.gcd(x, y)
    ans = g if kind == "HCF" else x * y // g
    q = f"Find {kind} of {x} and {y}."
    return q, str(ans), _choices_with_distractors(ans, spread=5)


def q_sit(diff: str):
    """Simple interest."""
    p = random.randint(1000, {"Easy": 8000, "Medium": 15000, "Hard": 30000}[diff])
    r = random.choice([4, 5, 6, 7, 8, 10, 12, 15])
    t = random.choice([1, 2, 3, 4, 5])
    si = p * r * t / 100
    q = f"Simple Interest on ₹{p} at {r}% p.a. for {t} years?"
    return q, str(int(si)), _choices_with_distractors(int(si), spread=100)


def q_sdt(diff: str):
    """Speed–Time–Distance triplets."""
    s = random.randint(20, {"Easy": 80, "Medium": 120, "Hard": 180}[diff])
    t = random.randint(1, 6)
    d = s * t
    kind = random.choice(["distance", "time", "speed"])
    if kind == "distance":
        q, ans = f"Speed {s} km/h for {t} h. Find distance (km).", d
    elif kind == "time":
        q, ans = f"Distance {d} km at {s} km/h. Find time (h).", round(d / s, 2)
    else:
        q, ans = f"Distance {d} km in {t} h. Find speed (km/h).", s
    return q, str(ans), _choices_with_distractors(ans, spread=5)


# A tiny chemistry bank to keep things fun.
CHEM = [
    ("Hydrogen", "H", 1), ("Helium", "He", 2), ("Lithium", "Li", 3), ("Beryllium", "Be", 4),
    ("Boron", "B", 5), ("Carbon", "C", 6), ("Nitrogen", "N", 7), ("Oxygen", "O", 8),
    ("Fluorine", "F", 9), ("Neon", "Ne", 10), ("Sodium", "Na", 11), ("Magnesium", "Mg", 12),
    ("Aluminium", "Al", 13), ("Silicon", "Si", 14), ("Phosphorus", "P", 15), ("Sulfur", "S", 16),
    ("Chlorine", "Cl", 17), ("Argon", "Ar", 18), ("Potassium", "K", 19), ("Calcium", "Ca", 20),
    ("Iron", "Fe", 26), ("Copper", "Cu", 29), ("Zinc", "Zn", 30), ("Silver", "Ag", 47), ("Gold", "Au", 79)
]


def q_chem(diff: str):
    """Ask for symbol or atomic number."""
    name, sym, num = random.choice(CHEM)
    if random.choice([True, False]):
        q, ans = f"Symbol of **{name}**?", sym
        choices = _choices_with_distractors(ans, spread=2)
        if len(choices) < 4:
            pool = [s for _, s, _ in CHEM]
            while len(choices) < 4:
                choices.append(random.choice(pool))
        random.shuffle(choices)
        return q, ans, choices
    else:
        q, ans = f"Atomic number of **{name}**?", str(num)
        return q, ans, _choices_with_distractors(ans, spread=2)


# Registry of topics → generator function (easier to extend).
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

# =========================
# AI Helpers for Exam Tab
# =========================

def _has_groq():
    """True if a plausible Groq key is present."""
    key = (os.getenv("GROQ_API_KEY") or "").strip()
    return key.startswith("gsk_")


def llm_explain_for_exam(topic, question_text, selected, correct):
    """Short explanation shown after a wrong answer."""
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
    """Optionally add a slightly harder follow-up after a correct answer."""
    if not _has_groq():
        return ""
    prompt = (
        "Create ONE slightly harder follow-up MCQ (A–D) on the same concept. "
        "Return exactly:\nQuestion\nA) ...\nB) ...\nC) ...\nD) ...\nAnswer: X\n\n"
        f"Topic: {topic}\nSeed question: {question_text}\nSeed answer: {correct}"
    )
    out = explain_with_groq(prompt)
    return "\n\n**Try this:**\n" + out if out else ""


# =========================
# Logging (CSV / Firestore)
# =========================

def _csv_write_header_if_needed(path: str, fields: list[str]):
    """Create CSV with header if it doesn't exist yet."""
    newfile = not os.path.exists(path)
    if newfile:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()


def _log_csv(row: dict):
    """Append one attempt to the CSV log."""
    fields = ["ts", "student", "session", "topic_index", "topic", "difficulty", "question",
              "answer", "selected", "is_correct", "topic_correct", "topic_attempts", "overall_correct"]
    _csv_write_header_if_needed(LOG_PATH, fields)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writerow(row)


_FS_CLIENT = None  # Firestore client (lazy-created)


def _get_fs_client():
    """Create or return a cached Firestore client if enabled and available."""
    global _FS_CLIENT
    if _FS_CLIENT is not None:
        return _FS_CLIENT
    if not (USE_FIRESTORE and firestore):
        return None
    try:
        project = os.getenv("FIRESTORE_PROJECT") or None
        _FS_CLIENT = firestore.Client(project=project)
        return _FS_CLIENT
    except Exception:
        return None


def _log_firestore(row: dict):
    """Best-effort log to Firestore."""
    client = _get_fs_client()
    if not client:
        return False
    try:
        client.collection(FS_COLLECTION).add(row)
        return True
    except Exception:
        return False


def log_attempt(row: dict):
    """Public logging function. Tries Firestore first (if enabled), then CSV."""
    ok = False
    if USE_FIRESTORE and firestore:
        ok = _log_firestore(row)
    if not ok:
        _log_csv(row)


# =========================
# UI Theme & CSS (Kid-friendly)
# =========================
kids_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="pink",
    neutral_hue="gray",
)
# Some gradio versions don't support setters — keep it optional.
try:
    kids_theme = kids_theme.set(
        body_text_size="17px",
        font=["ui-sans-serif", "system-ui", "Segoe UI", "Arial"],
    )
except TypeError:
    pass

KIDS_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@600;700&family=Poppins:wght@400;600&display=swap');
:root { --brand-1:#6D5EF7; --brand-2:#F471B5; --bg-1:#f7f8ff; --card-bg:#ffffffd9; }
html, body, .gradio-container {
  background: radial-gradient(1200px 600px at 10% 0%, #fef3ff 0%, #f0f8ff 30%, #eef2ff 60%, #ffffff 100%) fixed;
}
.hero { margin: 8px 0 18px 0; background: linear-gradient(120deg, #a5b4fc 0%, #fbcfe8 50%, #fde68a 100%);
  border-radius: 28px; padding: 28px 22px; box-shadow: 0 8px 24px rgba(109, 94, 247, 0.15); }
.hero h1 { font-family: 'Baloo 2','Poppins',system-ui,sans-serif; font-size: 42px; margin: 0 0 6px 0; color: #1f2937; }
.hero .subtitle { font-family: 'Poppins',system-ui,sans-serif; font-size: 18px; color:#374151; }
.hero .badge { display:inline-block; background:#fff; color:#4f46e5; font-weight:700; border-radius:999px; padding:6px 12px; margin-top:8px;
  box-shadow:0 6px 14px rgba(0,0,0,0.06); }
.kid-card { background: var(--card-bg); border-radius: 22px; box-shadow: 0 10px 22px rgba(31, 41, 55, 0.08); padding: 14px; border: 1px solid rgba(99,102,241,0.10); }
.kid-section-title { font-weight: 700; font-size: 20px; color: #1f2937; }
.kid-btn .btn { font-weight: 700; border-radius: 999px; padding: 10px 18px; }
.kid-primary .btn { background: linear-gradient(90deg, var(--brand-1), var(--brand-2)) !important; color:#fff !important; border:none !important; }
.kid-radio .wrap-inner { gap: 10px; }
.kid-radio input[type="radio"] + label { background:#fff; border:2px solid #e5e7eb !important; border-radius:999px; padding:8px 14px; box-shadow:0 4px 10px rgba(0,0,0,0.04); }
.kid-radio input[type="radio"]:checked + label { border-color:#6366f1 !important; background:#eef2ff; }
.kid-progress { background:#fff; border-radius:14px; padding:8px 12px; border:1px dashed #c7d2fe; }
"""

# Gradio containers differ across versions; pick one that exists.
if hasattr(gr, "Box"):
    Container = gr.Box
elif hasattr(gr, "Column"):
    Container = gr.Column
else:
    Container = gr.Group

# =========================
# Build UI with Gradio Blocks
# =========================
with gr.Blocks(theme=kids_theme, css=KIDS_CSS, title="Vidya Setu — Tutor + Adaptive Exam") as demo:
    # Header banner
    gr.HTML("""
    <div class="hero">
      <div class="badge">✨ Kid-Friendly • Safe • Fun</div>
      <h1>🎒 Vidya Setu</h1>
      <div class="subtitle">Learn • Practice • Shine</div>
    </div>
    """)

    # ----- Tab 1: Tutor -----
    with gr.Tab("Tutor (Groq)"):
        with Container(elem_classes=["kid-card"]):
            q = gr.Textbox(label="📝 Ask a question")
            want_hint = gr.Checkbox(label="💡 Hint only")
            ans = gr.Textbox(label="🤖 Tutor answer")
            dbg = gr.Textbox(label="🔧 Debug/State")
            gr.Button("Send", elem_classes=["kid-btn", "kid-primary"]).click(tutor, [q, want_hint], [ans, dbg])

    # ----- Tab 2: Adaptive Exam -----
    with gr.Tab("Adaptive Exam"):
        # Setup section
        with Container(elem_classes=["kid-card"]):
            gr.Markdown(
                f"### 🚀 How it works\nEach topic continues until you get **{TOPIC_TARGET_CORRECT} correct**. "
                "Difficulty adapts up/down. Final report at the end.",
                elem_classes=["kid-section-title"],
            )
            student_id = gr.Textbox(label="👧👦 Student ID (for progress tracking)", placeholder="e.g., 2025A001")
            topic_select = gr.CheckboxGroup(
                choices=list(TOPICS.keys()),
                value=["Fractions", "Decimals", "Percentages"],
                label="🎯 Select topics (in order)",
            )
            start_btn = gr.Button("Start Exam ▶", elem_classes=["kid-btn", "kid-primary"])

        # Question + actions
        with Container(elem_classes=["kid-card"]):
            status = gr.Markdown()
            question_md = gr.Markdown()
            options = gr.Radio(choices=[], label="Choose your answer", elem_classes=["kid-radio"])
            with gr.Row():
                check_btn = gr.Button("Check ✅", elem_classes=["kid-btn", "kid-primary"])
                next_btn = gr.Button("Next ▶", elem_classes=["kid-btn"])
            feedback = gr.Markdown()

        # Progress strip
        with gr.Row():
            with Container(elem_classes=["kid-card", "kid-progress"]):
                curr_topic = gr.Markdown()
                progress = gr.Markown()
                score_box = gr.Markdown()

        # Report + CSV path
        report = gr.Markdown()
        with Container(elem_classes=["kid-card"]):
            csv_path_show = gr.Textbox(value=LOG_PATH, label="📁 CSV path", interactive=False)
            report_file = gr.File(label="⬇️ Download Final Report", interactive=False)
            download_btn = gr.Button("Refresh CSV Path", elem_classes=["kid-btn"])

        # Shared state object for the whole exam flow
        exam_state = gr.State({})

        # ---------- Inner helpers bound to UI ----------

        def gen_question(topic: str, diff_idx: int):
            """Generate one (q, ans, choices, difficulty-name) tuple for the chosen topic."""
            diff = DIFF_LEVELS[diff_idx]
            q, ans, choices = TOPICS[topic](diff)
            return q, ans, choices, diff

        def adapt(up: bool, idx: int) -> int:
            """Up/down difficulty bounded in [0, 2]."""
            return min(2, idx + 1) if up else max(0, idx - 1)

        def start_exam(student, topics):
            """Initialize a new session and load the first question."""
            if not topics:
                return ("Please select at least one topic.", "", gr.update(choices=[], value=None),
                        "", "", "", {}, "", LOG_PATH, None)
            st = {
                "student": (student or "anon").strip(),
                "session": str(uuid.uuid4())[:8],
                "topics": topics,
                "topic_idx": 0,
                "diff_idx": 0,  # start Easy
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
            curr = f"**Topic:** {topic}  |  **Difficulty:** {diff}"
            prog = f"Correct in topic: **0 / {TOPIC_TARGET_CORRECT}**"
            score = f"**Overall correct:** 0"
            return status, q, gr.update(choices=choices, value=None), curr, prog, score, st, "", LOG_PATH, None

        def check_answer(choice, st):
            """Evaluate the chosen option, log the attempt, and show feedback/LLM help."""
            if not st or st.get("answer") is None:
                return "Click **Start Exam** first.", st, "", LOG_PATH, None

            topic = st["topics"][st["topic_idx"]]
            st["attempts_in_topic"] += 1
            st["results"][topic]["attempts"] += 1

            is_correct = (choice is not None) and (str(choice).strip() == str(st["answer"]).strip())

            # Prepare a CSV/Firestore row for analytics.
            row = {
                "ts": datetime.datetime.utcnow().isoformat(),
                "student": st.get("student", "anon"),
                "session": st.get("session", ""),
                "topic_index": st["topic_idx"],
                "topic": topic,
                "difficulty": DIFF_LEVELS[st["diff_idx"]],
                "question": st.get("last_question", ""),
                "answer": st["answer"],
                "selected": (choice or ""),
                "is_correct": int(is_correct),
                "topic_correct": st["correct_in_topic"] + (1 if is_correct else 0),
                "topic_attempts": st["attempts_in_topic"],
                "overall_correct": st["score_total"] + (1 if is_correct else 0),
            }
            try:
                log_attempt(row)  # never break the UI if logging fails
            except Exception:
                pass

            # Adapt difficulty and craft feedback.
            if is_correct:
                st["correct_in_topic"] += 1
                st["results"][topic]["correct"] += 1
                st["score_total"] += 1
                follow = llm_followup(topic, st.get("last_question", ""), st["answer"])
                msg = "✅ Correct!" + (f"{follow}" if follow else "")
                st["diff_idx"] = adapt(True, st["diff_idx"])
            else:
                ai_note = llm_explain_for_exam(topic, st.get("last_question", ""), choice, st["answer"])
                msg = f"❌ Wrong. Correct answer: **{st['answer']}**" + (f"{ai_note}" if ai_note else "")
                st["diff_idx"] = adapt(False, st["diff_idx"])

            st["last_feedback"] = msg
            return msg, st, "", LOG_PATH, None

        def next_step(st):
            """
            Move to the next question OR next topic OR finalize the exam.
            Also writes the final Markdown report to REPORT_DIR.
            """
            if not st or "topics" not in st:
                return ("Click **Start Exam** first.", "", gr.update(choices=[], value=None),
                        "", "", "", st, "", LOG_PATH, None)

            topics = st["topics"]
            topic = topics[st["topic_idx"]]

            # Did we hit the target for this topic?
            if st["correct_in_topic"] >= TOPIC_TARGET_CORRECT:
                st["topic_idx"] += 1
                st["diff_idx"] = 0
                st["correct_in_topic"] = 0
                st["attempts_in_topic"] = 0

                # If all topics complete → build the final report.
                if st["topic_idx"] >= len(topics):
                    lines = [
                        f"### 📊 Final Report — overall correct: **{st['score_total']}**  "
                        f"|  Student: **{st.get('student','anon')}**  |  Session: `{st.get('session','')}`"
                    ]
                    weak = []
                    for t, v in st["results"].items():
                        att = max(1, v["attempts"])
                        acc = round(100 * v["correct"] / att)
                        lines.append(f"- **{t}** — {v['correct']} / {v['attempts']}  (**{acc}%**)")
                        if acc < 80:
                            weak.append((t, acc))
                    if weak:
                        lines.append("\n**Focus on:** " + ", ".join([f\"{t} ({a}%)\" for t, a in weak]))
                    if os.getenv("GROQ_API_KEY"):
                        lines.append("\n_Status: GROQ key present — AI hints & follow-ups enabled._")
                    report_md = "\n".join(lines)

                    # Save for download
                    os.makedirs(REPORT_DIR, exist_ok=True)
                    fname = f"report_{st.get('student','anon')}_{st.get('session','')}.md"
                    report_path = os.path.join(REPORT_DIR, fname)
                    with open(report_path, "w", encoding="utf-8") as f:
                        f.write(report_md)

                    return ("🎉 Exam finished!", "", gr.update(choices=[], value=None),
                            "", "", f"**Overall correct:** {st['score_total']}", st, report_md, LOG_PATH, report_path)

                # Next topic, first question
                topic = topics[st["topic_idx"]]
                q, ans, choices, diff = gen_question(topic, st["diff_idx"])
                st["answer"] = ans
                st["last_question"] = q
                curr = f"**Next topic:** {topic}  |  **Difficulty:** {diff}"
                prog = f"Correct in topic: **0 / {TOPIC_TARGET_CORRECT}**"
                score = f"**Overall correct:** {st['score_total']}"
                return ("➡️ Topic complete. Moving on.", q, gr.update(choices=choices, value=None),
                        curr, prog, score, st, "", LOG_PATH, None)

            # Same topic → just load another question.
            q, ans, choices, diff = gen_question(topic, st["diff_idx"])
            st["answer"] = ans
            st["last_question"] = q
            curr = f"**Topic:** {topic}  |  **Difficulty:** {diff}"
            prog = f"Correct in topic: **{st['correct_in_topic']} / {TOPIC_TARGET_CORRECT}**"
            score = f"**Overall correct:** {st['score_total']}"
            return (st.get("last_feedback", ""), q, gr.update(choices=choices, value=None),
                    curr, prog, score, st, "", LOG_PATH, None)

        # Wire UI → handlers (keep the SAME state object everywhere)
        start_btn.click(
            start_exam, [student_id, topic_select],
            [status, question_md, options, curr_topic, progress, score_box, exam_state, report, csv_path_show, report_file],
        )
        check_btn.click(
            check_answer, [options, exam_state],
            [feedback, exam_state, report, csv_path_show, report_file],
        )
        next_btn.click(
            next_step, [exam_state],
            [feedback, question_md, options, curr_topic, progress, score_box, exam_state, report, csv_path_show, report_file],
        )
        download_btn.click(lambda: LOG_PATH, [], [csv_path_show])

# =========================
# FastAPI app (Gradio mounted)
# =========================
app = FastAPI()

@app.get("/healthz")
def healthz():
    """Simple liveness endpoint for platforms like Cloud Run."""
    return {"ok": True}

# Host the Gradio UI at "/"
app = gr.mount_gradio_app(app, demo, path="/")
