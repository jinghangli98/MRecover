FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY mrecover/ mrecover/

RUN printf "torch==2.2.2\n" > /tmp/constraints.txt \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==2.2.2" \
    && pip install --no-cache-dir --constraint /tmp/constraints.txt .

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/cache/huggingface

ENTRYPOINT ["mrecover"]
