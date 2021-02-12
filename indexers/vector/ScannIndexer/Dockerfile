FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /

RUN apt-get update && apt-get upgrade -y && apt-get clean

# get python 3.7, default for (Ubuntu 18.04) is Python 3.6
RUN apt-get install -y curl python3.7 python3.7-dev python3.7-distutils
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1

# Use python 3.7 as default
RUN update-alternatives --set python /usr/bin/python3.7

# get pip
RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py --force-reinstall && \
    rm get-pip.py

# Scann needs gcc9 and default for (Ubuntu 18.04) is gcc7
RUN apt update && apt install -y build-essential && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt update && \
    apt install -y gcc-9 g++-9


# get the scann whl
RUN curl https://storage.googleapis.com/scann/releases/1.0.0/scann-1.0.0-cp37-cp37m-linux_x86_64.whl --output scann-1.0.0-cp37-cp37m-linux_x86_64.whl

COPY requirements.txt /requirements.txt

RUN pip install -r requirements.txt

COPY . /workspace
WORKDIR /workspace

# for testing the image
RUN pip install pytest && pytest -v -s

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]