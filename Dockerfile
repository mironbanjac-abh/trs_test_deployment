FROM python:3.12

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

## docker build -t arima-service .
## docker run -p 8000:8000 arima-service
