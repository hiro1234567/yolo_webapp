FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# OpenCV などで必要になる最低限のライブラリ
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

# 依存ライブラリ
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# アプリ本体をコピー
COPY . .

# Cloud Run などで PORT 環境変数を使えるようにしておく（デフォルト8080）
ENV PORT=8080

# FastAPI + Uvicorn 起動
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
