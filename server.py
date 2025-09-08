import os, traceback, requests
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, RedirectResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
import gradio as gr

# Gradio is mounted behind a proxy at /tutor (important for Railway)
os.environ.setdefault("GRADIO_ROOT_PATH", "/tutor")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Helpful startup logs (you'll see these in Railway logs)
print(f"[startup] BASE_DIR={BASE_DIR}")
print(f"[startup] STATIC_DIR exists? {os.path.isdir(STATIC_DIR)}")
if os.path.isdir(STATIC_DIR):
    try:
        print(f"[startup] static contains: {os.listdir(STATIC_DIR)}")
    except Exception as _e:
        print("[startup] can't list static/:", _e)

# Try to import your Gradio Blocks 'demo'
try:
    from app_groq_practice import demo as _demo  # expects: with gr.Blocks() as demo:
    demo = _demo
    print("[startup] Loaded app_groq_practice.demo OK")
except Exception as e:
    err = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
    print("[startup] Failed to import app_groq_practice.demo")
    print(err)
    # Fallback UI so the server still starts
    with gr.Blocks(title="Startup Error") as demo:
        gr.Markdown("### ❌ Failed to load `app_groq_practice.demo`\nYour server is up, but the app import failed.")
        gr.Textbox(value=err, label="Import error details", lines=20, interactive=False)
        gr.Markdown("- Make sure the file **app_groq_practice.py** exists at the repo root and defines `demo`.")

# ------------------ FastAPI ------------------
app = FastAPI()

# /static (index.html, signup.html, …)
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", include_in_schema=False)
def root():
    p = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(p):
        return FileResponse(p)
    return PlainTextResponse("static/index.html not found", status_code=404)

@app.get("/signup", include_in_schema=False)
def signup():
    p = os.path.join(STATIC_DIR, "signup.html")
    if os.path.isfile(p):
        return FileResponse(p)
    return PlainTextResponse("static/signup.html not found", status_code=404)

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    p = os.path.join(STATIC_DIR, "favicon.ico")
    if os.path.isfile(p):
        return FileResponse(p)
    return PlainTextResponse("", status_code=204)

# Mount Gradio at /tutor
app = gr.mount_gradio_app(app, demo, path="/tutor")

# Redirect bare queue → Gradio queue (fixes WS pathing)
@app.api_route("/queue/{rest:path}", methods=["GET", "POST", "OPTIONS"])
async def gradio_queue_shim(rest: str, request: Request):
    return RedirectResponse(url=f"/tutor/queue/{rest}", status_code=307)

@app.get("/ping")
def ping():
    return {"ok": True}

# Debug: check environment & static
@app.get("/_debug")
def _debug():
    key = (os.getenv("GROQ_API_KEY") or "").strip().strip('"').strip("'")
    return {
        "has_groq_key": bool(key),
        "groq_key_prefix": key[:4],
        "groq_key_len": len(key),
        "GRADIO_ROOT_PATH": os.getenv("GRADIO_ROOT_PATH"),
        "static_exists": os.path.isdir(STATIC_DIR),
        "static_files": os.listdir(STATIC_DIR) if os.path.isdir(STATIC_DIR) else [],
    }

# Debug: call Groq directly from the server
@app.get("/_groq_check")
def _groq_check():
    key = (os.getenv("GROQ_API_KEY") or "").strip().strip('"').strip("'")
    hdr = {"Authorization": f"Bearer {key}"}
    try:
        r = requests.get("https://api.groq.com/openai/v1/models", headers=hdr, timeout=10)
        ct = r.headers.get("content-type", "")
        body = r.json() if "application/json" in ct else r.text[:300]
        return {"status": r.status_code, "ok": r.status_code == 200, "body": body}
    except Exception as e:
        return {"status": None, "ok": False, "error": str(e)}
