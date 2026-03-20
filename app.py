from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")


class PredictRequest(BaseModel):
    # Teacher/demo-friendly shape (single row): {"features": [..30 floats..]}
    features: Optional[List[float]] = Field(
        default=None,
        description="Single example feature vector (length 30 for breast cancer dataset)",
    )

    # Batch-friendly shape (multiple rows): {"instances": [[..],[..]]}
    instances: Optional[List[List[float]]] = Field(
        default=None,
        description="2D array: (n_samples, n_features)",
    )


class PredictResponse(BaseModel):
    prediction: Optional[int] = None
    predictions: List[int]
    probabilities: Optional[List[float]] = None


app = FastAPI(title="Breast Cancer Classifier", version="1.0")

_model = None


def _read_sample_request_text() -> str:
        try:
                p = Path("sample_request.json")
                if p.exists():
                        return p.read_text(encoding="utf-8")
        except Exception:
                pass
        return (
                '{\n'
                '  "features": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]\n'
                '}\n'
        )


_SAMPLE_REQUEST_TEXT = _read_sample_request_text()


@app.get("/", response_class=HTMLResponse)
def simple_ui() -> str:
        sample_js = json.dumps(_SAMPLE_REQUEST_TEXT)
        return f"""<!doctype html>
<html lang=\"en\">
    <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <title>Breast Cancer API</title>
    </head>
    <body>
        <h2>Breast Cancer API</h2>
        <p>Use this page to call <code>/health</code> and <code>/predict</code>.</p>

        <div>
            <button id=\"btnHealth\">GET /health</button>
            <button id=\"btnPredict\">POST /predict</button>
            <button id=\"btnClear\">Clear output</button>
        </div>

        <h3>Request body (JSON)</h3>
        <textarea id=\"body\" rows=\"10\" cols=\"100\" spellcheck=\"false\"></textarea>

        <h3>Terminal output</h3>
        <pre id=\"log\" style=\"white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; border: 1px solid #ccc; padding: 8px;\"></pre>

        <script>
            const SAMPLE_BODY = {sample_js};
            const logEl = document.getElementById('log');
            const bodyEl = document.getElementById('body');
            bodyEl.value = SAMPLE_BODY;

            function log(line) {{
                logEl.textContent += line + "\n";
            }}

            async function callJson(method, path, bodyText) {{
                const url = path;
                log(`> ${{method}} ${{url}}`);

                const opts = {{ method }};
                if (bodyText !== undefined) {{
                    opts.headers = {{ 'Content-Type': 'application/json' }};
                    opts.body = bodyText;
                }}

                const resp = await fetch(url, opts);
                const text = await resp.text();
                log(`< ${{resp.status}} ${{resp.statusText}}`);
                log(text);
                log('');
            }}

            document.getElementById('btnHealth').addEventListener('click', async () => {{
                try {{
                    await callJson('GET', '/health');
                }} catch (e) {{
                    log(String(e));
                }}
            }});

            document.getElementById('btnPredict').addEventListener('click', async () => {{
                try {{
                    // Basic JSON validation so errors are clearer
                    JSON.parse(bodyEl.value);
                    await callJson('POST', '/predict', bodyEl.value);
                }} catch (e) {{
                    log(String(e));
                }}
            }});

            document.getElementById('btnClear').addEventListener('click', () => {{
                logEl.textContent = '';
            }});
        </script>
    </body>
</html>
"""


@app.on_event("startup")
def _load_model() -> None:
    global _model
    # Best-practice/class requirement: load from the dummy model registry folder.
    # Backward compatible fallback: if the registry file doesn't exist, try ./model.pkl.
    try_paths = [MODEL_PATH]
    if MODEL_PATH != "model.pkl":
        try_paths.append("model.pkl")

    last_err: Optional[Exception] = None
    for p in try_paths:
        try:
            _model = joblib.load(p)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e

    raise RuntimeError(
        f"Failed to load model. Tried: {try_paths}. Last error: {last_err}"
    )


@app.get("/health")
def health() -> dict:
    # Keep this response minimal to match the class demo expectation.
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if _model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if (req.features is None) == (req.instances is None):
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of: 'features' (single row) or 'instances' (batch)",
        )

    if req.features is not None:
        X = np.asarray([req.features], dtype=float)
    else:
        X = np.asarray(req.instances, dtype=float)

    if X.ndim != 2:
        raise HTTPException(status_code=400, detail="instances must be a 2D array")

    # For sklearn breast cancer dataset this is always 30 features.
    if X.shape[1] != 30:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 30 features per row, got {X.shape[1]}",
        )

    preds = _model.predict(X)
    predictions = [int(x) for x in preds.tolist()]

    probabilities = None
    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(X)
        if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] >= 2:
            probabilities = [float(x) for x in proba[:, 1].tolist()]

    single = len(predictions) == 1
    return PredictResponse(
        prediction=predictions[0] if single else None,
        predictions=predictions,
        probabilities=probabilities,
    )
