# RetinaTriage inference API
#
# CPU-only build: the CUDA torch wheels are several GB, and the demo runs
# comfortably on CPU. Swap the extra index URL below for a GPU deployment.
FROM python:3.11-slim

# TORCH_HOME keeps the torch hub cache inside /app, where the non-root user
# that this image finally runs as can still read it.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RETINATRIAGE_FORCE_CPU=1 \
    TORCH_HOME=/app/.torch

WORKDIR /app

# libgl / libglib are needed by opencv, even the headless build.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first, so code edits don't invalidate the (large) torch layer.
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Bake the ImageNet backbone into the image so the first request isn't a
# multi-hundred-megabyte download.
RUN python -c "from torchvision import models; models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)"

# Run as an unprivileged user.
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app /home/appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
