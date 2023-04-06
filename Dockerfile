# FROM nvcr.io/nvidia/tensorflow:21.03-tf2-py3
# check if this version has memory leak, looks like it doens't leak, or not as significant as it was before
FROM nvcr.io/nvidia/tensorflow:22.01-tf2-py3

RUN apt-get update && apt install screen -y

COPY ./requirements.txt /tmp/docker-build-files/requirements.txt

WORKDIR /tmp/docker-build-files/

RUN /usr/bin/python -m pip install -r requirements.txt

WORKDIR /
