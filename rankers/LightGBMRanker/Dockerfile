FROM jinaai/jina as base

RUN apt-get update && apt-get install libgomp1

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install the third-party requirements
RUN pip install -r requirements.txt

# for testing the image
FROM base
RUN pip install pytest sklearn && pytest -v -s

FROM base
RUN pip install pytest sklearn && \
    python -c 'from tests.test_lightgbmranker import _pretrained_model; import os; os.mkdir("tmp"); model_path = os.path.join("tmp", "model.txt"); _pretrained_model(model_path)' && \
    pip uninstall -y sklearn pytest
ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]
