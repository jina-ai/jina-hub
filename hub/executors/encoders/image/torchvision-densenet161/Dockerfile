FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel


RUN pip install jina
RUN python -c "from torchvision.models import densenet161; model_name = densenet161(pretrained=True, progress=True)"

ADD *.py *.yml ./

ENTRYPOINT ["jina", "pod", "--yaml-path", "ImageTorchEncoder"]
