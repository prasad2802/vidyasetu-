import os, time, random, requests, sqlite3, json, string
import gradio as gr
import sympy as sp
from datetime import datetime
import secrets
import hashlib

# =============================================================
# APP CONFIG
# =============================================================
APP_NAME = "📘 Personalized Tutor + Practice (Groq)"
DB_PATH = os.getenv("AUTH_DB", "users.db")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
TOPIC_TARGET_CORRECT = 5

# ==== Groq model fallback order (edit to your taste) ====
GROQ_FALLBACKS = [
    os.getenv("GROQ_MODEL", "").strip() or "llama-3.3-70b-versatile",
    "gemma2-9b-it",
    "llama-3.1-8b-instant",
    "deepseek-r1-distill-llama-70b",
    "groq.compound-mini",
    "allam-2-7b",
]
_LAST_WORKING_MODEL = None

# =========================
# Global state for tutor
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

# =============================================================
# AUTH LAYER (SQLite + salted PBKDF2 hash)
# =============================================================

def db_conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    with db_conn() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS users(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                name TEXT,
                pw_hash TEXT NOT NULL,
                pw_salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        con.commit()


def _hash_password(password: str, salt_hex: str | None = None) -> tuple[str, str]:
    """Return (hash_hex, salt_hex). Uses PBKDF2-HMAC-SHA256."""
    if salt_hex is None:
        salt_hex = secrets.token_hex(16)
    salt = bytes.fromhex(salt_hex)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return dk.hex(), salt_hex


def create_user(email: str, password: str, name: str = "") -> tuple[bool, str]:
    email = email.strip().lower()
    if not email or "@" not in email:
        return False, "Please enter a valid email."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    try:
        pw_hash, salt_hex = _hash_password(password)
        with db_conn() as con:
            con.execute(
                "INSERT INTO users(email, name, pw_hash, pw_salt, created_at) VALUES(?,?,?,?,?)",
                (email, name.strip(), pw_hash, salt_hex, datetime.utcnow().isoformat()),
            )
            con.commit()
        return True, "✅ Account created. You can sign in now."
    except sqlite3.IntegrityError:
        return False, "This email is already registered."
    except Exception as e:
        return False, f"Error creating user: {e}"


def authenticate_user(email: str, password: str) -> tuple[bool, str | dict]:
    email = email.strip().lower()
    try:
        with db_conn() as con:
            cur = con.execute("SELECT id, email, name, pw_hash, pw_salt FROM users WHERE email=?", (email,))
            row = cur.fetchone()
            if not row:
                return False, "No account found. Please sign up."
            uid, email, name, pw_hash_db, salt_hex = row
            calc_hash, _ = _hash_password(password, salt_hex)
            if secrets.compare_digest(calc_hash, pw_hash_db):
                return True, {"id": uid, "email": email, "name": name or email.split("@")[0]}
            else:
                return False, "Incorrect password."
    except Exception as e:
        return False, f"Auth error: {e}"


# =============================================================
# Groq caller (with automatic fallback)
# =============================================================

def explain_with_groq(prompt: str) -> str:
    global _LAST_WORKING_MODEL

    key = os.getenv("GROQ_API_KEY") or ""
    if not key.startswith("gsk_"):
        return "❌ GROQ_API_KEY missing/invalid. Set once:  setx GROQ_API_KEY \"gsk_...\"  and reopen PowerShell."

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

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


# =============================================================
# Math Guard
# =============================================================

def solve_math(expr: str):
    try:
        expr = expr.replace("=", "")
        res = sp.simplify(expr)
        return str(res)
    except Exception:
        return None


# =============================================================
# Personalization
# =============================================================

def predict_next_prob():
    return 0.5 * STATE["rolling_acc_user"] + 0.5 * STATE["rolling_acc_skill"]


def adapt_mode(p):
    if p < 0.4:
        return "scaffold"
    elif p < 0.7:
        return "normal"
    else:
        return "challenge"


# =============================================================
# Tutor core
# =============================================================

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


# =============================================================
# MCQ BANK (Math + Chemistry)
# =============================================================
MCQ_BANK = {
    "Fractions": [
        {"Q": "What is 1/2 + 1/5?", "A": "7/10", "B": "6/7", "C": "3/5", "D": "9/10", "Correct": "A", "Expl": "Common denominator 10 → (5+2)/10 = 7/10."},
        {"Q": "Simplify 4/8", "A": "1/4", "B": "1/2", "C": "2/3", "D": "3/4", "Correct": "B", "Expl": "Divide numerator & denominator by 4 → 1/2."},
        {"Q": "(2/3) × (3/5) = ?", "A": "6/15", "B": "2/5", "C": "3/10", "D": "5/6", "Correct": "B", "Expl": "(2×3)/(3×5) = 6/15 = 2/5."},
        {"Q": "Which is larger?", "A": "3/4", "B": "2/3", "C": "5/8", "D": "1/2", "Correct": "A", "Expl": "3/4=0.75; 2/3≈0.667; 5/8=0.625; 1/2=0.5."},
        {"Q": "What is 3/4 − 1/8?", "A": "5/8", "B": "2/3", "C": "1/2", "D": "3/8", "Correct": "A", "Expl": "LCM 8 → 6/8 − 1/8 = 5/8."},
        {"Q": "Simplify 9/12", "A": "3/4", "B": "2/3", "C": "1/3", "D": "3/5", "Correct": "A", "Expl": "Divide by 3 → 3/4."},
    ],
    "Decimals": [
        {"Q": "0.25 + 0.5 = ?", "A": "0.55", "B": "0.65", "C": "0.75", "D": "0.85", "Correct": "C", "Expl": "0.25 + 0.50 = 0.75."},
        {"Q": "0.8 × 0.2 = ?", "A": "0.16", "B": "0.12", "C": "0.18", "D": "0.20", "Correct": "A", "Expl": "8×2=16; two decimal places → 0.16."},
        {"Q": "Round 5.678 to 2 decimals", "A": "5.68", "B": "5.67", "C": "5.70", "D": "5.60", "Correct": "A", "Expl": "Third digit 8→ round 5.67 up to 5.68."},
        {"Q": "6.3 ÷ 0.9 = ?", "A": "0.7", "B": "7", "C": "6.9", "D": "0.14", "Correct": "B", "Expl": "Scale ×10: 63 ÷ 9 = 7."},
        {"Q": "Which is greater?", "A": "0.25", "B": "0.205", "C": "0.2", "D": "0.024", "Correct": "A", "Expl": "0.25 > 0.205 > 0.2 > 0.024."},
        {"Q": "0.4 × 0.06 = ?", "A": "0.024", "B": "0.24", "C": "0.0024", "D": "0.026", "Correct": "A", "Expl": "4×6=24; total 3 decimal places → 0.024."},
    ],
    "Percents": [
        {"Q": "10% of 250 = ?", "A": "25", "B": "2.5", "C": "0.25", "D": "2500", "Correct": "A", "Expl": "10% = 0.1; 0.1×250=25."},
        {"Q": "Increase 120 by 25%", "A": "145", "B": "150", "C": "155", "D": "125", "Correct": "B", "Expl": "120×1.25 = 150."},
        {"Q": "Convert 45% to decimal", "A": "0.45", "B": "4.5", "C": "0.045", "D": "45", "Correct": "A", "Expl": "45/100 = 0.45."},
        {"Q": "Decrease 80 by 15%", "A": "72", "B": "68", "C": "70", "D": "78", "Correct": "B", "Expl": "80×0.85 = 68."},
        {"Q": "What is 30% of 90?", "A": "27", "B": "30", "C": "21", "D": "33", "Correct": "A", "Expl": "0.30×90=27."},
        {"Q": "Which is larger?", "A": "35%", "B": "0.34", "C": "3/10", "D": "32%", "Correct": "A", "Expl": "35% = 0.35; others < 0.35."},
    ],
    "Geometry (Triangles)": [
        {"Q": "In a right triangle, a=3, b=4. Hypotenuse?", "A": "5", "B": "6", "C": "7", "D": "4.5", "Correct": "A", "Expl": "Pythagoras: √(3²+4²)=√25=5."},
        {"Q": "Sum of interior angles of a triangle?", "A": "90°", "B": "180°", "C": "270°", "D": "360°", "Correct": "B", "Expl": "Always 180°."},
        {"Q": "An isosceles triangle has…", "A": "all equal sides", "B": "two equal sides", "C": "no equal sides", "D": "right angle", "Correct": "B", "Expl": "Isosceles ⇒ exactly two equal sides."},
        {"Q": "Area of right triangle (legs 6,8)?", "A": "24", "B": "28", "C": "30", "D": "48", "Correct": "A", "Expl": "Area = ½×base×height = ½×6×8 = 24."},
        {"Q": "If sides are 5,12,13, triangle is…", "A": "acute", "B": "right", "C": "obtuse", "D": "equilateral", "Correct": "B", "Expl": "5²+12²=25+144=169=13² → right."},
        {"Q": "Which can be triangle sides?", "A": "2,3,6", "B": "2,2,3", "C": "1,1,3", "D": "4,1,2", "Correct": "B", "Expl": "Triangle inequality holds only for 2,2,3."},
    ],
    "Algebra (Basics)": [
        {"Q": "Solve: x + 5 = 12", "A": "x=5", "B": "x=7", "C": "x=12", "D": "x=17", "Correct": "B", "Expl": "Subtract 5 both sides → x=7."},
        {"Q": "Solve: 3x = 21", "A": "x=9", "B": "x=7", "C": "x=6", "D": "x=63", "Correct": "B", "Expl": "Divide both sides by 3 → x=7."},
        {"Q": "Value of 2x+3 when x=4", "A": "11", "B": "10", "C": "8", "D": "15", "Correct": "A", "Expl": "2×4+3 = 8+3 = 11."},
        {"Q": "Simplify: 5x + 2x", "A": "7", "B": "7x", "C": "x^7", "D": "5x^2", "Correct": "B", "Expl": "Like terms add: 5x+2x=7x."},
        {"Q": "Solve: x−3=−1", "A": "x=2", "B": "x=−2", "C": "x=1", "D": "x=4", "Correct": "A", "Expl": "Add 3 to both sides → x=2."},
        {"Q": "If y=2x and x=5, y=?", "A": "7", "B": "10", "C": "2", "D": "25", "Correct": "B", "Expl": "y=2×5=10."},
    ],
    "Ratios & Proportion": [
        {"Q": "Ratio 6:9 simplified", "A": "2:3", "B": "3:2", "C": "1:3", "D": "3:1", "Correct": "A", "Expl": "Divide both by 3 → 2:3."},
        {"Q": "If 2/3 = x/9, x=?", "A": "3", "B": "4", "C": "5", "D": "6", "Correct": "D", "Expl": "Cross-multiply: 2×9=18; 18/3=6."},
        {"Q": "Proportion true?", "A": "4/6 = 2/3", "B": "5/8 = 3/5", "C": "2/7 = 1/4", "D": "3/9 = 2/3", "Correct": "A", "Expl": "4/6 simplifies to 2/3."},
        {"Q": "Divide ₹60 in ratio 2:3", "A": "₹24 and ₹36", "B": "₹20 and ₹40", "C": "₹30 and ₹30", "D": "₹10 and ₹50", "Correct": "A", "Expl": "Total parts 5; each part 12; 2×12=24, 3×12=36."},
        {"Q": "Scale a recipe 2×: 3 cups flour becomes…", "A": "5 cups", "B": "6 cups", "C": "4 cups", "D": "3 cups", "Correct": "B", "Expl": "Double of 3 is 6."},
        {"Q": "Which ratio equals 0.5?", "A": "3:5", "B": "1:3", "C": "1:2", "D": "2:1", "Correct": "C", "Expl": "1:2 = 0.5."},
    ],
    # Chemistry
    "Chemistry (Matter)": [
        {"Q": "Matter is anything that…", "A": "has mass and occupies space", "B": "is visible only", "C": "has no volume", "D": "has no mass", "Correct": "A", "Expl": "Definition of matter."},
        {"Q": "Which is NOT a state of matter?", "A": "Solid", "B": "Liquid", "C": "Gas", "D": "Light", "Correct": "D", "Expl": "Light is energy, not matter."},
        {"Q": "Particles in solids are…", "A": "far apart", "B": "free to flow", "C": "closely packed", "D": "ionized", "Correct": "C", "Expl": "Solids have tightly packed particles."},
        {"Q": "Change from liquid to gas is…", "A": "Condensation", "B": "Evaporation", "C": "Freezing", "D": "Sublimation", "Correct": "B", "Expl": "Liquid → gas = evaporation/boiling."},
        {"Q": "Sublimation is…", "A": "solid→gas", "B": "gas→solid", "C": "liquid→solid", "D": "liquid→gas", "Correct": "A", "Expl": "Direct solid to gas (e.g., dry ice)."},
        {"Q": "A mixture that looks uniform is…", "A": "heterogeneous", "B": "homogeneous", "C": "colloidal only", "D": "compound", "Correct": "B", "Expl": "Homogeneous looks uniform (e.g., salt water)."},
    ],
    "Chemistry (Elements & Symbols)": [
        {"Q": "Symbol for Hydrogen", "A": "H", "B": "He", "C": "Hy", "D": "Hg", "Correct": "A", "Expl": "Hydrogen = H."},
        {"Q": "Symbol for Sodium", "A": "So", "B": "Na", "C": "Sn", "D": "Sd", "Correct": "B", "Expl": "Sodium = Na (from ‘Natrium’)."},
        {"Q": "Symbol for Potassium", "A": "P", "B": "Pt", "C": "K", "D": "Po", "Correct": "C", "Expl": "Potassium = K (from ‘Kalium’)."},
        {"Q": "Symbol for Iron", "A": "Ir", "B": "Fe", "C": "In", "D": "I", "Correct": "B", "Expl": "Iron = Fe (from ‘Ferrum’)."},
        {"Q": "Symbol for Carbon", "A": "C", "B": "Ca", "C": "Co", "D": "Cr", "Correct": "A", "Expl": "Carbon = C."},
        {"Q": "Symbol for Oxygen", "A": "O", "B": "Ox", "C": "Og", "D": "Oy", "Correct": "A", "Expl": "Oxygen = O."},
    ],
    "Chemistry (Periodic Table Basics)": [
        {"Q": "Atomic number is…", "A": "protons in nucleus", "B": "neutrons in nucleus", "C": "electrons in outer shell", "D": "mass of atom", "Correct": "A", "Expl": "Atomic number = number of protons."},
        {"Q": "Element with atomic number 1", "A": "Helium", "B": "Hydrogen", "C": "Lithium", "D": "Oxygen", "Correct": "B", "Expl": "H has Z=1."},
        {"Q": "Group 1 elements are called…", "A": "Halogens", "B": "Noble gases", "C": "Alkali metals", "D": "Metalloids", "Correct": "C", "Expl": "Group 1 = alkali metals."},
        {"Q": "Noble gases are in…", "A": "Group 17", "B": "Group 18", "C": "Period 2 only", "D": "Transition block", "Correct": "B", "Expl": "Group 18 = noble gases (He, Ne, Ar…)."},
        {"Q": "Chlorine’s symbol", "A": "Cl", "B": "Ch", "C": "Cr", "D": "Cn", "Correct": "A", "Expl": "Chlorine = Cl."},
        {"Q": "Across a period, atomic size generally…", "A": "increases", "B": "decreases", "C": "same", "D": "random", "Correct": "B", "Expl": "Effective nuclear charge ↑ → size ↓."},
    ],
}

# Practice session state
P_SESS = {"topic": None, "pool": [], "idx": 0, "attempts": 0, "corrects": 0}


# =============================================================
# Practice logic
# =============================================================

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
    is_right = choice and choice.upper().strip() == correct

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


# =============================================================
# UI: Sign-In / Sign-Up + Gated App
# =============================================================

def do_signup(email, name, password, confirm):
    if password != confirm:
        return "❌ Passwords do not match."
    ok, msg = create_user(email, password, name)
    return msg


def do_login(email, password):
    ok, data = authenticate_user(email, password)
    if ok:
        # store a basic session token (ephemeral)
        token = secrets.token_urlsafe(16)
        session = {"user": data, "token": token, "ts": time.time()}
        return json.dumps(session), f"✅ Welcome, {data['name']}!"
    else:
        return "", f"❌ {data}"


def do_logout():
    return "", "👋 Logged out."


# toggle main app visibility after successful login

def show_app_if_logged(session_json):
    try:
        session = json.loads(session_json or "{}")
        if session.get("user") and session.get("token"):
            return gr.update(visible=True), f"Signed in as: {session['user']['email']}"
    except Exception:
        pass
    return gr.update(visible=False), "Not signed in"


# hook tutor

def on_tutor(q, hint):
    if not q:
        return "Please type a question.", ""
    return tutor(q, hint)


# hook practice start/check

def on_start(topic):
    return start_practice(topic)


def on_check(choice):
    if not choice:
        return "Choose A/B/C/D first.", "", ""
    return check_answer(choice)


# Build DB at boot
init_db()

# ============ GRADIO APP ============
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")) as demo:
    gr.Markdown(f"# {APP_NAME}\nSecure access + website-ready")

    # STATE
    session_state = gr.State("")  # holds JSON for {user, token}

    with gr.Row():
        signed_banner = gr.Markdown("Not signed in")

    with gr.Tab("Sign In"):
        with gr.Row():
            with gr.Column(scale=1):
                login_email = gr.Textbox(label="Email", placeholder="you@example.com")
                login_pass = gr.Textbox(label="Password", type="password")
                login_btn = gr.Button("Sign In", variant="primary")
                login_msg = gr.Markdown()
                login_btn.click(
                    do_login,
                    [login_email, login_pass],
                    [session_state, login_msg],
                ).then(
                    show_app_if_logged,
                    [session_state],
                    [app_groq_practice, signed_banner],
                )

    with gr.Tab("Sign Up"):
        with gr.Row():
            with gr.Column(scale=1):
                su_name = gr.Textbox(label="Name (optional)")
                su_email = gr.Textbox(label="Email")
                su_pass = gr.Textbox(label="Password (min 6 chars)", type="password")
                su_conf = gr.Textbox(label="Confirm Password", type="password")
                su_btn = gr.Button("Create Account", variant="primary")
                su_msg = gr.Markdown()
                su_btn.click(do_signup, [su_email, su_name, su_pass, su_conf], [su_msg])

    # Gated main app (hidden until login)
    with gr.Group(visible=False) as app_groq_practice:
        gr.Markdown("## 🎓 Tutor Portal")
        with gr.Accordion("Account", open=False):
            logout = gr.Button("Log out")
            # Clear session and update visibility/banner
            logout.click(lambda: "", outputs=[session_state]).then(
                show_app_if_logged, [session_state], [app_groq_practice, signed_banner]
            )

        with gr.Tab("Tutor"):
            q = gr.Textbox(label="Ask a question", placeholder="e.g., Solve 2x+5=11 or What is photosynthesis?")
            hint = gr.Checkbox(label="Hint only")
            ans = gr.Textbox(label="Tutor answer")
            dbg = gr.Textbox(label="Debug/State")
            gr.Button("Send", variant="primary").click(on_tutor, [q, hint], [ans, dbg])

        with gr.Tab("Practice"):
            topic = gr.Dropdown(choices=list(MCQ_BANK.keys()), value="Fractions", label="Topic")
            out1 = gr.Textbox(label="Status/Feedback")
            out2 = gr.Textbox(label="Question")
            out3 = gr.Textbox(label="Progress")
            start = gr.Button("Start Practice", variant="primary")
            start.click(on_start, [topic], [out1, out2, out3])
            choice = gr.Dropdown(choices=["A", "B", "C", "D"], label="Your Answer")
            check = gr.Button("Check Answer")
            check.click(on_check, [choice], [out1, out2, out3])

    # Show correct area depending on session
    demo.load(show_app_if_logged, [session_state], [app_groq_practice, signed_banner])

# Entry point (Spaces, local, etc.)
if __name__ == "__main__":
    # HINT: Set your key first (Windows PowerShell):
    #   setx GROQ_API_KEY "gsk_..."  ; then reopen shell
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
