FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .
COPY agent ./agent
COPY api ./api
COPY training ./training
COPY dags ./dags
COPY data ./data
COPY dvc.yaml params.yaml ./
RUN python -m training.train
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
