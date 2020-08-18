FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

RUN pip install transformers --no-cache-dir --compile && \
    python -c "from transformers import *; x='bert-base-uncased'; BertModel.from_pretrained(x); BertTokenizer.from_pretrained(x)"

RUN pip install jina

ENTRYPOINT ["jina", "pod", "--yaml-path", "TransformerTorchEncoder"]