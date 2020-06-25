FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

RUN pip install jina

RUN python -c "from torchvision.models import resnext50_32x4d; model_name = resnext50_32x4d(pretrained=True, progress=True, **kwargs)"

ADD *.py *.yml ./

ENTRYPOINT["jina", "pod", "--yaml-path", "ImageTorchEncoder"]