# server.py
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import gradio as gr

# 1) Import the Blocks instance named "demo"
from app_groq_practice import demo

app = FastAPI()

# 2) Serve static assets under /static and make "/" return index.html
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", include_in_schema=False)
def root():
    return FileResponse("static/index.html")

# 3) Mount Gradio at /tutor (works across Gradio versions)
#    Newer Gradio returns the FastAPI app; older modifies in place.
try:
    # Gradio >= 4.x
    app = gr.mount_gradio_app(app, demo, path="/tutor")
except TypeError:
    # Very old Gradio fallback
    gr.mount_gradio_app(app, demo, path="/tutor")

# Optional: simple health check
@app.get("/ping")
def ping():
    return {"ok": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
