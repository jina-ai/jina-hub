FROM jinaai/jina:1.0.16

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install the third-party requirements
RUN pip install -r requirements.txt

# for testing the image
RUN pip install pytest && pytest && rm -rf '/root/.cache/huggingface/transformers/' && rm -rf /tmp/{*,.*}

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]