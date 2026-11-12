FROM python:3.11-slim

RUN apt-get update && apt-get upgrade -y && rm -rf /var/cache/apt/

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rbg_planner/ rbg_planner/
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

ENTRYPOINT ["rbg-planner"]
