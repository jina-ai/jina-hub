FROM jinaai/jina

# build sptag
WORKDIR /app
RUN apt-get update && \
    apt-get -y install wget python3-dev python3-pip git swig libboost1.67 libboost1.67-dev

RUN wget https://cmake.org/files/v3.15/cmake-3.15.5.tar.gz && \
    tar zvxf cmake-3.15.5.tar.gz && \
    cd cmake-3.15.5 && \
    ./bootstrap && \
    make -j2 && \
    make install

RUN git clone https://github.com/microsoft/SPTAG.git
RUN cd SPTAG && \
    git fetch origin a91ed49a041a24bd145874d3b3c815bde0430e65 && \
    git reset --hard FETCH_HEAD && \
    mkdir build && \
    cd build && \
    cmake .. && \
    grep -rl python2.7 CMakeCache.txt | xargs sed -i 's/python2.7/python3.7/g' && \
    make -j2 && \
    mv ../Release /app

# so python can find the SPTAG module
ENV PYTHONPATH=/app/Release

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install the third-party requirements
RUN pip install -r requirements.txt

# for testing the image
RUN pip install pytest && pytest

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]