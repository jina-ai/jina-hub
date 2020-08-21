FROM jinaai/jina:devel

# build sptag
WORKDIR /app
#RUN apt-get -y update && \
#    apt-get -y upgrade && \
#    apt-get install -y git && \
RUN apt-get update && apt-get -y install wget build-essential python3-dev python3-pip git swig

RUN git clone https://github.com/microsoft/SPTAG.git .

#COPY CMakeLists.txt ./
#COPY AnnService ./AnnService/
#COPY Test ./Test/
#COPY Wrappers ./Wrappers/

    # remove the following if you don't want to build the wrappers

# cmake >= 3.12 is required
RUN wget "https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.tar.gz" -q -O - \
        | tar -xz --strip-components=1 -C /usr/local

# specific version of boost
RUN wget "https://dl.bintray.com/boostorg/release/1.67.0/source/boost_1_67_0.tar.gz" -q -O - \
        | tar -xz && \
        cd boost_1_67_0 && \
        ./bootstrap.sh && \
        ./b2 install && \
        # update ld cache so it finds boost in /usr/local/lib
        ldconfig && \
        cd .. && rm -rf boost_1_67_0

# build
RUN mkdir build && cd build && cmake .. && make && cd ..

# so python can find the SPTAG module
ENV PYTHONPATH=/app/Release

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install the third-party requirements
RUN pip install -r requirements.txt

# for testing the image
RUN pip install pytest && pytest

# ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]