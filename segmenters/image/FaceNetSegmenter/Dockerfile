FROM jinaai/jina AS base

# Setup the workspace
COPY . /workspace
WORKDIR /workspace

# Install dependencies
RUN pip install -r requirements.txt

# Run tests
FROM base
RUN pip install pytest && pytest

# Run encoder as pod
FROM base
ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]
