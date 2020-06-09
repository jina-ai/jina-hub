FROM jinaai/jina:devel

RUN pip install nmslib
RUN python -c "import nmslib;"

RUN pip install jina

ENTRYPOINT ["jina", "pod", "--yaml-path", "NmslibIndexer"]