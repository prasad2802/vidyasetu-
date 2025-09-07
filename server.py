# server.py  (hardened)
import os, sys, traceback
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, RedirectResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
import gradio as gr

# --- Gradio mount lives at /tutor (important on Railway) ---
os.environ.setdefault("GRADIO_ROOT_PATH", "/tutor")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Minimal preflight logging so you see it in Railway logs
print(f"[startup] BASE_DIR={BASE_DIR}")
print(f"[startup] STATIC_DIR exists? {os.path.isdir(STATIC_DIR)}")
if os.path.isdir(STATIC_DIR):
    print(f"[startup] static contains: {os.listdir(STATIC_DIR)}")
else:
    print("[startup] !! static/ folder missing at deploy root")

# ----- Try to load your real Blocks() from app_groq_practice.py -----
demo = None
import_error_text = ""

try:
    from app_groq_practice import demo as _demo  # your real Blocks
    demo = _demo
    print("[startup] Loaded app_groq_practice.demo OK")
except Exception as e:
    import_error_text = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
    print("[startup] Failed to import app_groq_practice.demo")
    print(import_error_text)

    # Fallback Gradio UI that shows the import error instead of crashing
    with gr.Blocks(title="Startup Error") as demo:
        gr.Markdown(
            "### ❌ Failed to load `app_groq_practice.demo`\n"
            "Your server is up, but the app import failed. See details below."
        )
        gr.Code(value=import_error_text, language="text", label="Import error details")
        gr.Markdown(
            "- Confirm the file is named **app_groq_practice.py** at the repo root.\n"
            "- Confirm it defines a top-level `with gr.Blocks() as demo:`."
        )

# ------------- Build FastAPI and mount everything -------------
app = FastAPI()

# serve /static and two friendly routes
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

# mount gradio at /tutor
app = gr.mount_gradio_app(app, demo, path="/tutor")

# redirect bare queue requests → gradio queue
@app.api_route("/queue/{rest:path}", methods=["GET", "POST", "OPTIONS"])
async def gradio_queue_shim(rest: str, request: Request):
    return RedirectResponse(url=f"/tutor/queue/{rest}", status_code=307)

@app.get("/ping")
def ping():
    return {"ok": True, "has_demo": demo is not None}

# Extra debug: list /app directory & env
@app.get("/_debug")
def debug():
    try:
        return {
            "cwd": os.getcwd(),
            "listdir": os.listdir("."),
            "static_exists": os.path.isdir(STATIC_DIR),
            "static_listdir": os.listdir(STATIC_DIR) if os.path.isdir(STATIC_DIR) else [],
            "GRADIO_ROOT_PATH": os.getenv("GRADIO_ROOT_PATH"),
            "PYTHONPATH": os.getenv("PYTHONPATH"),
        }
    except Exception as e:
        return {"error": str(e)}
