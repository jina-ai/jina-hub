FROM conda/miniconda3-centos7:latest

WORKDIR /

RUN conda update -n base conda && \
    conda install -y -c pytorch faiss-cpu

RUN python -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir jina[http]

RUN python -c "import faiss;index = faiss.IndexFlatL2(10)"

COPY faiss.yml .

ENTRYPOINT ["jina", "pod", "--yaml-path", "faiss.yml"]