FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

RUN apt update
RUN apt upgrade -y
RUN apt install -y build-essential
RUN apt install -y libsndfile1

WORKDIR /workspace
COPY requirements.txt requirements.txt
COPY post_requirements.txt post_requirements.txt
COPY src src
COPY data/cache data/cache
COPY data/chordlab data/chordlab
COPY data/index.csv data/index.csv
COPY data/index_big.csv data/index_big.csv

RUN pip install -r requirements.txt
RUN pip install -r post_requirements.txt
