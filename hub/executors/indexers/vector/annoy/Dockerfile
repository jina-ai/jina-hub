FROM jinaai/jina:devel

RUN pip install annoy
RUN python -c "from annoy import AnnoyIndex;"

RUN pip install jina

ENTRYPOINT ["jina", "pod", "--yaml-path", "NmslibIndexer"]