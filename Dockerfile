FROM python:3.10-slim

# Make Python logs appear immediately and keep bytecode caches out of the image.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STUDENT_HEALTH_PROJECT_ROOT=/app

WORKDIR /app

# LightGBM needs the OpenMP runtime provided by libgomp1.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies before source code so Docker can cache this slow layer.
COPY requirements.lock ./
RUN pip install --no-cache-dir --requirement requirements.lock

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

# Include only the two production artifacts used by the API.
COPY artifacts/lightgbm_pipeline.joblib ./artifacts/
COPY artifacts/label_encoder.joblib ./artifacts/

# Run the service without root privileges.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "student_health.api:app", "--host", "0.0.0.0", "--port", "8000"]
