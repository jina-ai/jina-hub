FROM jinaai/jina:devel

RUN pip install jina
RUN python -c "import faiss;"

ENTRYPOINT ["jina", "pod", "--yaml-path", "FaissIndexer"]