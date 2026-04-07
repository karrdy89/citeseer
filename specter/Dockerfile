FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# 빌드 시 모델 미리 다운로드
RUN python -c "\
from transformers import AutoTokenizer; \
from adapters import AutoAdapterModel; \
tok = AutoTokenizer.from_pretrained('allenai/specter2_base'); \
m = AutoAdapterModel.from_pretrained('allenai/specter2_base'); \
m.load_adapter('allenai/specter2_adhoc_query', source='hf', load_as='q', set_active=True); \
print('Model downloaded')"

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
