import os, traceback
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, RedirectResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
import gradio as gr

# Ensure Gradio knows it's mounted at /tutor when behind a proxy
os.environ.setdefault("GRADIO_ROOT_PATH", "/tutor")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Minimal preflight logs (helpful on Railway)
print(f"[startup] BASE_DIR={BASE_DIR}")
print(f"[startup] STATIC_DIR exists? {os.path.isdir(STATIC_DIR)}")
if os.path.isdir(STATIC_DIR):
    print(f"[startup] static contains: {os.listdir(STATIC_DIR)}")
else:
    print("[startup] !! static/ folder missing at deploy root")

# --- Try to import your Gradio Blocks() named `demo` ---
demo = None
try:
    from app_groq_practice import demo as _demo  # expects with gr.Blocks() as demo:
    demo = _demo
    print("[startup] Loaded app_groq_practice.demo OK")
except Exception as e:
    import_error_text = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
    print("[startup] Failed to import app_groq_practice.demo")
    print(import_error_text)

    # Fallback UI (doesn't crash Gradio if import fails)
    with gr.Blocks(title="Startup Error") as demo:
        gr.Markdown(
            "### ❌ Failed to load `app_groq_practice.demo`\n"
            "Your server is up, but the app import failed. See details below."
        )
        gr.Textbox(
            value=import_error_text,
            label="Import error details",
            lines=20,
            interactive=False
        )
        gr.Markdown(
            "- Confirm the file is named **app_groq_practice.py** at the repo root.\n"
            "- Confirm it defines a top-level `with gr.Blocks() as demo:`."
        )

# ------------ FastAPI app + static files ------------
app = FastAPI()

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", include_in_schema=False)
def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return PlainTextResponse("static/index.html not found", status_code=404)

@app.get("/signup", include_in_schema=False)
def signup():
    signup_path = os.path.join(STATIC_DIR, "signup.html")
    if os.path.isfile(signup_path):
        return FileResponse(signup_path)
    return PlainTextResponse("static/signup.html not found", status_code=404)

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    fav = os.path.join(STATIC_DIR, "favicon.ico")
    if os.path.isfile(fav):
        return FileResponse(fav)
    return PlainTextResponse("", status_code=204)

# Mount Gradio at /tutor
app = gr.mount_gradio_app(app, demo, path="/tutor")

# Redirect bare queue endpoint → Gradio queue path
@app.api_route("/queue/{rest:path}", methods=["GET", "POST", "OPTIONS"])
async def gradio_queue_shim(rest: str, request: Request):
    return RedirectResponse(url=f"/tutor/queue/{rest}", status_code=307)

@app.get("/ping")
def ping():
    return {"ok": True, "has_demo": demo is not None}

@app.get("/_debug")
def debug():
    key = (os.getenv("GROQ_API_KEY") or "").strip().strip('"').strip("'")
    return {
        "has_groq_key": bool(key),
        "groq_key_prefix": key[:4],   # should be "gsk_"
        "groq_key_len": len(key),
        "static_exists": os.path.isdir(STATIC_DIR),
        "static_listdir": os.listdir(STATIC_DIR) if os.path.isdir(STATIC_DIR) else [],
        "GRADIO_ROOT_PATH": os.getenv("GRADIO_ROOT_PATH"),
    }

# No __main__ block needed on Railway; it uses the start command.

