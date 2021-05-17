FROM jinaai/jina:1.2.1 as base

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install the third-party requirements
RUN pip install -r requirements.txt

# for training a model
RUN python train_default_model.py

# for testing the image
FROM base
RUN pip install pytest && pytest

FROM base
ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]
