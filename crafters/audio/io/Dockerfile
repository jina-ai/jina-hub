FROM jinaai/jina:devel

ADD * ./
RUN pip install -r requirements.txt && python -m unittest test.py

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]