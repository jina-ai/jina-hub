FROM jinaai/jina

# setup the workspace
COPY . /workspace
WORKDIR /workspace

RUN python3 -m pip install --upgrade pip
# install the third-party requirements
RUN pip install -r requirements.txt

# for testing the image
RUN pip install pytest && pytest

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]