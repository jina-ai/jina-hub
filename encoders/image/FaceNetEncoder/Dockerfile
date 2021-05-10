FROM jinaai/jina:1.2.1 AS base

# Setup the workspace
COPY . /workspace
WORKDIR /workspace

# Install dependencies
RUN pip install -r requirements.txt

# Add vggface2 and casia-webface weights for the face embedder
RUN python -c "\
from facenet_pytorch import InceptionResnetV1; \
InceptionResnetV1(pretrained='vggface2').eval(); \
InceptionResnetV1(pretrained='casia-webface').eval() \
"

# Run tests
FROM base
RUN pip install pytest && pytest

# Run encoder as pod
FROM base
ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]
