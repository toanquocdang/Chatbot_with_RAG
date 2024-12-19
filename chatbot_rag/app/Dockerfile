# FROM nvidia/cuda:11.6.2-runtime-ubuntu18.04
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
FROM python:3.9.6

WORKDIR /app
COPY . /app
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
ENV PYTHONPATH=/app

CMD ["python", "app.py"]