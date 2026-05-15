FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04
LABEL description="Optional development Dockerfile for WMT26 Model Compression"
LABEL maintainer="WMT26 Model Compression Task Organizers"
LABEL version="1.0"
LABEL date="2025-05-15"

ARG DEBIAN_FRONTEND=noninteractive

# Docker is optional for WMT26; the official participant contract is setup.sh/run.sh.

# Install default packages
RUN apt update && apt upgrade --fix-missing -y
RUN apt install -y git python3 python3-pip emacs-nox vim wget curl ncdu tmux htop tree && \
    apt clean && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir --upgrade pip

# install torch built against the CUDA version of docker image; e.g. 12.8
# RUN python3 -m pip install --no-cache-dir torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
WORKDIR /work/wmt26-model-compression
COPY requirements.txt pyproject.toml README.md ./
COPY modelzip/ modelzip/
COPY setup.sh run.sh ./
RUN ls -lh && python3 -m pip install --no-cache-dir -e ./
RUN python3 -m modelzip.setup -h
##==============================================

# Copy model files with execution scripts
# Note: these models are for demonstration purposes only
# Do not include these in the submission image, include your compressed model(s) instead

#COPY workdir/models/gemma-3-12b-it-bnb-8bit /model/bnb-8bit
#COPY workdir/models/gemma-3-12b-it-bnb-4bit /model/bnb-4bit

# 
#RUN bash /model/bnb-8bit/run.sh ces-deu 1 <<< "This is a test with the 8-bit model."
#RUN bash /model/bnb-4bit/run.sh ces-deu 1 <<< "This is a test with the 4-bit model."
