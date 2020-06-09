FROM jinaai/jina:devel

RUN pip install jina
RUN python -c "import nmslib;"

ENTRYPOINT ["jina", "pod", "--yaml-path", "NmslibIndexer"]