# Bean: web app + API in one container (Hugging Face Spaces, Docker SDK)
FROM python:3.11-slim

# system libraries OpenCV needs
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH MPLCONFIGDIR=/tmp/mpl PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user app ./app
COPY --chown=user km ./km
COPY --chown=user pipeline ./pipeline

# warm up the OCR engine at build time so the first visitor isn't slow
RUN python -c "from pipeline.ticks import get_ocr, get_ocr_small; get_ocr(); get_ocr_small()"

EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
