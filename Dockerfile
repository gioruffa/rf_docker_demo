FROM python:3.6-slim-stretch
RUN mkdir /app && mkdir -p /data/out /data/in

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install -r requirements.txt

WORKDIR /app
COPY train_simple.py train_simple.py
ENTRYPOINT ["python", "train_simple.py"]
