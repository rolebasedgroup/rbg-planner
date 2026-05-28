FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rbg_profiler/ rbg_profiler/
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

ENTRYPOINT ["rbg-profiler"]
