FROM pytorch/pytorch:latest

RUN pip install jina
RUN python -c "import torch.nn as nn; import torch.nn.init as init"

ENTRYPOINT ["jina", "pod", "--yaml-path", "OnnxImageEncoder"]

