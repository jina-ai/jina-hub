FROM pytorch/manylinux-cuda100:latest

RUN pip install transformers --no-cache-dir --compile && \
    python -c "from transformers import *; x='hfl/chinese-roberta-wwm-ext'; BertModel.from_pretrained(x); BertTokenizer.from_pretrained(x)"

RUN pip install jina

ENV TMP_WORKSPACE /tmp/jina/webqa/
ENV BATCH_SIZE 32

ADD *.py *.yml ./

ENTRYPOINT ["jina", "pod", "--yaml-path", "roberta.yml"]