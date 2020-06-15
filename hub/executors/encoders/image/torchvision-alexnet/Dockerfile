FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

RUN pip install jina
RUN python -c "from torchvision.models import alexnet; model_name = alexnet(pretrained=True, progress=True, **kwargs)"

ADD *.py *.yml ./

ENTRYPOINT["jina", "pod", "--yaml-path", "ImageTorchEncoder"]
