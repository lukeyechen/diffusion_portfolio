FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
ENV APP_MODE=streamlit
EXPOSE 8080

CMD ["sh", "-c", "if [ \"${APP_MODE}\" = \"api\" ]; then exec uvicorn api:app --host=0.0.0.0 --port=${PORT}; else exec streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT} --server.headless=true; fi"]
