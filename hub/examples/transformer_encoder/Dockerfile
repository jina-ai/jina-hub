FROM jinaai/jina:master-debian

ENV TMP_WORKSPACE /tmp/jina/webqa/

RUN pip install transformers tensorflow torch torchvision --no-cache-dir --compile && \
python -c "from transformers import *; x='distilbert-base-uncased'; DistilBertModel.from_pretrained(x); DistilBertTokenizer.from_pretrained(x)"

ADD encoder.yml ./

ENTRYPOINT ["jina", "pod", "--yaml_path", "encoder.yml", "--timeout_ready", "60000"]