FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

RUN apt update
RUN apt upgrade -y
RUN apt install -y build-essential git ffmpeg libsndfile1 rubberband-cli

WORKDIR /workspace

# copy data
COPY data/chordlab data/chordlab
COPY data/index.csv data/index.csv

# copy and install requirements
COPY requirements.txt requirements.txt
COPY post_requirements.txt post_requirements.txt
RUN pip install -r requirements.txt
RUN pip install -r post_requirements.txt

# copy source code
COPY .git .git
COPY src src
