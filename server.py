# server.py
import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import gradio as gr

# 1) Import the Blocks instance named "demo"
from app_groq_practice import demo

app = FastAPI()

@app.api_route("/queue/{rest:path}", methods=["GET", "POST", "OPTIONS"])
async def _gradio_queue_shim(rest: str, request: Request):
    # Preserve method with 307 for POSTs
    return RedirectResponse(url=f"/tutor/queue/{rest}", status_code=307)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
# 2) Serve static assets under /static and make "/" return index.html
app.mount("/static", StaticFiles(directory="static"), name="static")
app = FastAPI()
@app.get("/ping")
def ping():
    return {"ok": True}
@app.get("/", include_in_schema=False)
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
@app.get("/signup", include_in_schema=False)
def signup():
    return FileResponse(os.path.join(STATIC_DIR, "signup.html"))

# 3) Mount Gradio at /tutor (works across Gradio versions)
#    Newer Gradio returns the FastAPI app; older modifies in place.
try:
    # Gradio >= 4.x
    app = gr.mount_gradio_app(app, demo, path="/tutor")
except TypeError:
    # Very old Gradio fallback
    gr.mount_gradio_app(app, demo, path="/tutor")
@app.api_route("/queue/{rest:path}", methods=["GET", "POST", "OPTIONS"])
async def gradio_queue_shim(rest: str, request: Request):
    return RedirectResponse(url=f"/tutor/queue/{rest}", status_code=307)
# Optional: simple health check
@app.get("/ping")
def ping():
    return {"ok": True}

if __name__ == "__main__":
    import os, uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))






