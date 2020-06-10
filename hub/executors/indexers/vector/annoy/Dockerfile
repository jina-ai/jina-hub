FROM jinaai/jina:devel

RUN apt-get update && apt-get install --no-install-recommends -y g++
RUN pip install annoy

RUN pip install jina

ENV TMP_WORKSPACE /tmp/jina/annoy/
ADD *.py *.yml ./

ENTRYPOINT ["jina", "pod", "--yaml-path", "annoy.yml"]