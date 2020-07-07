FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

ADD distilbert.yml ./

RUN pip install transformers --no-cache-dir --compile && \
    python -c "from transformers import *; x='distilbert-base-cased'; DistilBertModel.from_pretrained(x); DistilBertTokenizer.from_pretrained(x)"

RUN pip install jina

ENTRYPOINT ["jina", "pod", "--yaml-path", "distilbert.yml"]