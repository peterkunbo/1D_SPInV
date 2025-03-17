FROM python:3.9-slim

WORKDIR /1DSPINV

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "SP_APP.py", "--server.address=0.0.0.0"]