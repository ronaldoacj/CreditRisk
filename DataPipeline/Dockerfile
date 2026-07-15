FROM python:3.11-slim

WORKDIR /app

COPY DataPipeline/ /app/
COPY common/ /app/common/

COPY DataPipeline/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# config.py, data_sanitization.py e abt_transform.py já existentes no projeto
CMD ["python", "abt_transform.py"]
