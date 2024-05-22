# syntax=docker/dockerfile:1

FROM nvcr.io/nvidia/tensorflow:24.04-tf2-py3
WORKDIR .
COPY . .
EXPOSE 8000
RUN pip install -r requirements.txt --progress-bar off
CMD ["python", "hot_cnn.py"]