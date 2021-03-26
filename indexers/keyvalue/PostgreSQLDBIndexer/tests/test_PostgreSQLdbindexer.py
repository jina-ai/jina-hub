import os
import shutil
import sys
from typing import Iterator

import numpy as np
import pytest

from jina import Document
from .. import PostgreSQLDBIndexer

from tests import get_documents

def doc_without_embedding(d):
    new_doc = Document()
    new_doc.CopyFrom(d)
    new_doc.ClearField('embedding')
    return new_doc

def test_postgress():
    with PostgreSQLDBIndexer(username='susana', password='pwd', database='python_test',
            table='sql') as postgres_indexer:
        #postgres_indexer.add(8, 23, 'test')
        docs = list(get_documents(chunks=0, same_content=False))
        info = [
            (doc.id, doc.embedding, doc_without_embedding(doc).SerializeToString())
            for doc in docs
        ]
        if info:
            ids, vecs, metas = zip(*info)

            postgres_indexer.add(ids, vecs, metas)
            postgres_indexer.delete(ids[0])