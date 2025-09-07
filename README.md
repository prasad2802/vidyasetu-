# Adaptive AI Tutor Chatbot (ASSISTments)

This starter kit helps you build an **adaptive tutor chatbot** that personalizes hints and challenge levels using signals learned from the **ASSISTments** dataset (student interaction logs).

## Folder Layout
- `content/` — place your syllabus chapters here (plain text files) to feed the retriever.
- `data/` — put the raw `assistments.csv` (or multiple CSVs) here.
- `models/` — trained models will be saved here (e.g., `baseline.pkl`).
- `scripts/` — preprocessing and training scripts.
- `app.py` — Gradio-based chatbot app using retrieval + adaptation.
- `requirements.txt` — install dependencies with `pip install -r requirements.txt`.

## Quick Start
1. (Optional) Create a virtual env: `python -m venv .venv && source .venv/bin/activate` (Linux/Mac) or `.venv\Scripts\activate` (Windows).
2. Install deps: `pip install -r requirements.txt`
3. Put your raw dataset CSV as `data/assistments.csv` (ASSISTments skill-builder format; at minimum columns like `user_id, problem_id, skill, correct, timestamp`).
4. Preprocess: `python scripts/prep_assistments.py`
5. Train baseline: `python scripts/train_baseline.py`
6. Add some study content to `content/` (e.g., `chapter_fractions.txt`).
7. Launch app: `python app.py`

> If you don't have the dataset yet, you can still run the app in a demo mode (simple rules).

## Notes
- The baseline model is a simple logistic regression predicting whether the next answer is correct. The chatbot uses this prediction to adapt difficulty (scaffold vs. challenge).
- You can later replace the baseline with a knowledge-tracing model (e.g., DKT) or sequence model.
