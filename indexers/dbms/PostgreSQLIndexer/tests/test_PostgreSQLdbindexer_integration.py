from jina import Document
from jina.flow import Flow
import numpy as np

d_embedding = np.array([1, 1, 1, 1, 1, 1, 1])
c_embedding = np.array([2, 2, 2, 2, 2, 2, 2])


def get_documents(chunks, same_content, nr=10, index_start=0, same_tag_content=None):
    next_chunk_id = nr + index_start
    for i in range(index_start, nr + index_start):
        with Document() as d:
            d.id = i
            if same_content:
                d.text = "hello world"
                d.embedding = d_embedding
            else:
                d.text = f"hello world {i}"
                d.embedding = np.random.random(d_embedding.shape)
            if same_tag_content:
                d.tags["tag_field"] = "tag data"
            elif same_tag_content is False:
                d.tags["tag_field"] = f"tag data {i}"
            for j in range(chunks):
                with Document() as c:
                    c.id = next_chunk_id
                    if same_content:
                        c.text = "hello world from chunk"
                        c.embedding = c_embedding
                    else:
                        c.text = f"hello world from chunk {j}"
                        c.embedding = np.random.random(d_embedding.shape)
                    if same_tag_content:
                        c.tags["tag field"] = "tag data"
                    elif same_tag_content is False:
                        c.tags["tag field"] = f"tag data {next_chunk_id}"
                next_chunk_id += 1
                d.chunks.append(c)
        yield d


def doc_without_embedding(d):
    new_doc = Document()
    new_doc.CopyFrom(d)
    new_doc.ClearField("embedding")
    return new_doc


def validate(resp):
    assert resp
    #print(resp)
    #print(resp)

f = Flow().add(uses='docker://pos')
docs = list(get_documents(chunks=0, same_content=False))

info = [
    (doc.id, doc.embedding, doc_without_embedding(doc).SerializeToString())
    for doc in docs
]

ids, vecs, metas = zip(*info)

with f:
    f.index(inputs=docs, on_done=validate)