FROM jinaai/jina:devel

RUN pip install faiss
RUN apt-get update && apt-get install --no-install-recommends -y libopenblas-dev

RUN pip install jina

ENV TMP_WORKSPACE /tmp/jina/faiss_flat/
ADD *.py *.yml ./

ENTRYPOINT ["jina", "pod", "--yaml-path", "faiss.yml"]