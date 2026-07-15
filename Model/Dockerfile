FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    
WORKDIR /app

COPY Model/ /app/
COPY common/ /app/common/
COPY Model/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# train.py e tune.py já existentes no projeto (usam config.py)
CMD ["python", "train.py"]
