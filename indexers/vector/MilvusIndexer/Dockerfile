FROM milvusdb/milvus:cpu-latest

COPY . /workspace
WORKDIR /workspace

RUN yum -y install epel-release && yum -y install python-pip && \
yum -y install gcc openssl-devel bzip2-devel libffi-devel zlib-devel make && \
cd ~ && wget https://www.python.org/ftp/python/3.7.2/Python-3.7.2.tgz && tar xzf Python-3.7.2.tgz && \
cd ~/Python-3.7.2 && ./configure --enable-optimizations && make altinstall

RUN pip3.7 install --upgrade pip && pip3.7 install -r requirements.txt

ENV LD_LIBRARY_PATH=:/var/lib/milvus/lib
RUN ../var/lib/milvus/bin/milvus_server -c ../var/lib/milvus/conf/milvus.yaml -d && pip3.7 install pytest && pytest && ../var/lib/milvus/scripts/stop_server.sh

ENTRYPOINT ["../var/lib/milvus/bin/milvus_server -c ../var/lib/milvus/conf/milvus.yaml -d && jina pod --uses config.yml"]