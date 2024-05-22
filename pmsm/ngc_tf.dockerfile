# syntax=docker/dockerfile:1

FROM nvcr.io/nvidia/tensorflow:24.04-tf2-py3
WORKDIR /app
COPY . .
EXPOSE 8000