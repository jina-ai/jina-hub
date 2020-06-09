FROM jinaai/jina:devel

RUN apt-get update && apt-get install --no-install-recommends -y g++
RUN pip install annoy

RUN pip install jina

ENTRYPOINT ["jina", "pod", "--yaml-path", "AnnoyIndexer"]