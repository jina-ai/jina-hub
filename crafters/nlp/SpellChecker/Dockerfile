FROM jinaai/jina:1.1.5 as base

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install the third-party requirements
RUN pip install -r requirements.txt

# for testing the image
FROM base
RUN pip install pytest && pytest -v -s

FROM base
RUN python pyngramspell/train.py
ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]