FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

RUN apt-get update && apt-get install --no-install-recommends -y git wget gcc

RUN wget http://www.cs.toronto.edu/~faghri/vsepp/runs.tar && wget http://www.cs.toronto.edu/~faghri/vsepp/vocab.tar && \
tar -xvf runs.tar && tar -xvf vocab.tar && rm -rf runs/coco* && rm -rf runs/f30k_vse0/ && rm -rf runs/f30k_order*/ && \
rm -rf runs/f30k_vse++/ && rm -rf runs/f30k_vse++_resnet* && rm -rf runs/f30k_vse++_vggfull_finetune/ && \
rm -rf vocab/coco* && rm -rf vocab/f8k* && rm -rf vocab/10crop*/ && rm -rf vocab/f30k_precomp* && \
rm -rf runs.tar && rm -rf vocab.tar

COPY . /workspace
WORKDIR /workspace

RUN pip install -r requirements.txt && \
python -c "import torchvision.models as models; model = getattr(models, 'vgg19')(pretrained=True).eval()" && \
python -c "import nltk; nltk.download('punkt')"

# hotfix: upstream issues, https://github.com/google/flax/issues/269#issue-619773070
RUN pip uninstall -y dataclasses

RUN pip install pytest && pytest -v -s

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]
