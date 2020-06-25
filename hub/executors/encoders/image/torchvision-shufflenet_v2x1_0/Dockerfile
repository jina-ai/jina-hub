FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

RUN pip install jina

RUN python -c "from torchvision.models import shufflenet_v2_x1_0; model_name = shufflenet_v2_x1_0(pretrained=True, progress=True, **kwargs)"

ADD *.py *.yml ./

ENTRYPOINT["jina", "pod", "--yaml-path", "ImageTorchEncoder"]