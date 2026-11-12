FROM python:3.11-slim

RUN apt-get update && apt-get upgrade -y && rm -rf /var/cache/apt/

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r requirements.txt

COPY rbg_profiler/ rbg_profiler/
COPY pyproject.toml .
RUN pip install --no-cache-dir .

ENTRYPOINT ["rbg-profiler"]
