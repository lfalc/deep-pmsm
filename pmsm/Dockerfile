# syntax=docker/dockerfile:1

FROM nvcr.io/nvidia/tensorflow:24.04-tf2-py3
ADD CA.crt /usr/local/share/ca-certificates/CA.crt
RUN echo CA.crt >> /etc/ca-certificates.conf
RUN chmod 644 /usr/local/share/ca-certificates/CA.crt && update-ca-certificates
WORKDIR .
COPY . .
EXPOSE 8000
RUN apt-get update && apt-get install -y python3.10
RUN pip install -r requirements.txt --progress-bar off
CMD ["/bin/bash", "-c", "source shell_scripts/launch.tcsh"]