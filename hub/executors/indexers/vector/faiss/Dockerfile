FROM jinaai/jina:devel

RUN pip install faiss
RUN apt-get update && apt-get install --no-install-recommends -y libopenblas-dev

RUN pip install jina

ENTRYPOINT ["jina", "pod", "--yaml-path", "FaissIndexer"]