FROM python:3.11-slim

WORKDIR /app

# Install system-level build tools (often needed by ML libs)
RUN apt-get update && apt-get install -y \
	build-essential \
	&& rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

RUN pip install --upgrade pip && \
	pip install --no-cache-dir -r requirements.txt

# Copy app code and the exported model artifact
COPY app.py .
COPY model.pkl .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
