FROM conda/miniconda3-centos7:latest

WORKDIR /

RUN conda update -n base conda && \
    conda install -y -c pytorch faiss-cpu

COPY faiss.yml .

RUN python -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir jina[devel]

ENTRYPOINT ["jina", "pod", "--yaml-path", "faiss.yml"]
