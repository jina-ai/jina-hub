FROM jinaai/jina

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install the third-party requirements
# RUN pip install -r requirements.txt

RUN apt-get update && apt-get install --no-install-recommends -y git wget sudo cmake make gnupg gcc-multilib g++ default-libmysqlclient-dev gfortran

RUN git clone https://github.com/milvus-io/milvus && \
cd ./milvus/core && git checkout 0.10.3 && bash ubuntu_build_deps.sh && ./build.sh -t Release
# for testing the image
#RUN pip install pytest && pip install docker\
# && pytest && pip uninstall docker

ENTRYPOINT ["jina", "pod"]
#ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]