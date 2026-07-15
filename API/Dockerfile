FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    
WORKDIR /app

COPY API/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --timeout=120 --retries=10 -r requirements.txt

COPY API/main.py /app/main.py
COPY API/db.py   /app/db.py

COPY Model/__init__.py /app/Model/__init__.py
COPY Model/predict.py  /app/Model/predict.py

# curl usado no healthcheck do docker-compose
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
