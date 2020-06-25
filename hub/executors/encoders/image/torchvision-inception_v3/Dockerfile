FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

RUN pip install jina
RUN pip install scipy
RUN python -c "from torchvision.models import inception_v3; model_name = inception_v3(pretrained=True, progress=False, aux_logits=True, transform_input=True, **kwargs)"

ADD *.py *.yml ./

ENTRYPOINT["jina", "pod", "--yaml-path", "ImageTorchEncoder"]