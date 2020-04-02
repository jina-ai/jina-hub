FROM jinaai/jina:latest

RUN pip install transformers --no-cache-dir --compile && \
    python -c "from transformers import *; x='distilbert-base-uncased'; DistilBertModel.from_pretrained(x); DistilBertTokenizer.from_pretrained(x)"

ENTRYPOINT ["jina", "pod", "--yaml_path", "TransformerTextEncoder"]