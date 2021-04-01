import os

import numpy as np
import pytest

from .. import JiebaSegmenter

cur_dir = os.path.dirname(os.path.abspath(__file__))
path_dict_file = os.path.join(cur_dir, 'dict.txt')


def test_jieba_segmenter():
    segmenter = JiebaSegmenter(mode='accurate')
    text = '今天是个大晴天！安迪回来以后，我们准备去动物园。'
    docs_chunks = segmenter.segment(np.stack([text, text]))
    assert len(docs_chunks) == 2
    for chunks in docs_chunks:
        assert len(chunks) == 14


def test_jieba_user_dir():
    segmenter = JiebaSegmenter()
    text = '今天是个大晴天！安迪回来以后，我们准备去动物园。thisisnotachineseword'
    docs_chunks = segmenter.segment(np.stack([text, text]))
    assert len(docs_chunks) == 2
    for chunks in docs_chunks:
        assert len(chunks) == 15

    segmenter = JiebaSegmenter(user_dict_file=path_dict_file)
    text = '今天是个大晴天！安迪回来以后，我们准备去动物园。thisisnotachineseword'
    docs_chunks = segmenter.segment(np.stack([text, text]))
    assert len(docs_chunks) == 2
    for chunks in docs_chunks:
        assert len(chunks) == 20


def test_jieba_user_dir_file_not_found():
    with pytest.raises(FileNotFoundError):
        JiebaSegmenter(user_dict_file='/this/path/does/not/exist.txt')
