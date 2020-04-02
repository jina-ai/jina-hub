FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

RUN pip install transformers --no-cache-dir --compile && \
    python -c "from transformers import *; x='distilbert-base-uncased'; DistilBertModel.from_pretrained(x); DistilBertTokenizer.from_pretrained(x)"

# jina is not public yet, so we add from the local host and install it
ADD src/jina /.src

RUN pip install /.src/.

ENTRYPOINT ["jina", "pod", "--yaml_path", "TransformerTextEncoder"]