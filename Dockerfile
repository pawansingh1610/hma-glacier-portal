FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgdal-dev gdal-bin libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["gunicorn", "app:app", "--timeout", "300", "--workers", "1", "--bind", "0.0.0.0:7860"]
