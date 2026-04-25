FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY pyproject.toml .
COPY mrecover/ mrecover/

RUN pip install --no-cache-dir .

ENV HF_HOME=/cache/huggingface
# HF_TOKEN is required to download the gated model.
# Pass it at runtime: docker run -e HF_TOKEN=your_token ...
ENV HF_TOKEN=""

ENTRYPOINT ["mrecover"]
