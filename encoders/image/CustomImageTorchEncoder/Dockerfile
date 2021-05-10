FROM jinaai/jina:1.2.1

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install the third-party requirements
RUN pip install -r requirements.txt

# for testing the image
RUN python -c "import os; os.mkdir('models'); import torch; torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)" && \
python -c "import torch; from torchvision.models import mobilenet_v2; net = mobilenet_v2(); net.load_state_dict(torch.load('/root/.cache/torch/hub/checkpoints/mobilenet_v2-b0353104.pth')); torch.save(net, 'models/mobilenet_v2.pth')"

RUN pip install pytest && pytest -v -s

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]