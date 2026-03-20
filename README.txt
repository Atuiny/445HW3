Assignment 3 (barebones)
========================

This repo trains a binary classifier on the sklearn built-in breast cancer dataset and serves predictions via a FastAPI app in Docker.


1) Python setup (Windows PowerShell)
-----------------------------------
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt


2) Train locally
---------------
python src/train.py


3) Run the API in Docker (two ways)
----------------------------------

A) Instructor-style (load a provided image tar)
----------------------------------------------
docker load -i champion-inference-api.tar
docker images
docker run --rm -p 8000:8000 champion-inference-api:latest

# Then open:
#   http://127.0.0.1:8000/      (simple UI)
#   http://127.0.0.1:8000/docs


B) Build locally from this repo
-------------------------------
docker build -t bc-inference:latest .
docker run --rm -p 8000:8000 bc-inference:latest

# Then open:
#   http://127.0.0.1:8000/      (simple UI)
#   http://127.0.0.1:8000/docs
