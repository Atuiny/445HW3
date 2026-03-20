import json
import shutil
from pathlib import Path

registry_path = Path("registry/champion.json")
deployment_dir = Path("deployment")

deployment_dir.mkdir(parents=True, exist_ok=True)

if not registry_path.exists():
	raise FileNotFoundError("registry/champion.json not found")

with open(registry_path, "r", encoding="utf-8") as f:
	champion = json.load(f)

if not isinstance(champion, dict):
	raise ValueError("registry/champion.json must be a JSON object")

model_path = None
outputs = champion.get("outputs")
if isinstance(outputs, dict):
	model_path = outputs.get("model")

if not model_path:
	model_path = champion.get("model_path")

if not model_path:
	raise KeyError(
		"registry/champion.json is missing required field: outputs.model (or legacy model_path)"
	)

model_source = Path(model_path)
if not model_source.exists():
	raise FileNotFoundError(f"Champion model not found at {model_source}")

app_source = None
for candidate in (Path("app.py"), Path("api/app.py")):
	if candidate.exists():
		app_source = candidate
		break

if app_source is None:
	raise FileNotFoundError("API app not found (expected app.py or api/app.py)")

# Copy champion model into deployment folder
shutil.copy2(model_source, deployment_dir / "model.pkl")
# Copy API app into deployment folder
shutil.copy2(app_source, deployment_dir / "app.py")
print("Deployment files prepared successfully.")
print(f"Champion model copied from: {model_source}")
print("Files ready inside deployment/")