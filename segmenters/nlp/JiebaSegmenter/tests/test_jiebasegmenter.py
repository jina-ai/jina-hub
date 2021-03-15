import os

import pytest

from .. import JiebaSegmenter

cur_dir = os.path.dirname(os.path.abspath(__file__))
path_dict_file = os.path.join(cur_dir, 'dict.txt')

def test_jieba_crafter():
    jieba_crafter = JiebaSegmenter(mode='accurate')
    text = '今天是个大晴天！安迪回来以后，我们准备去动物园。'
    crafted_chunk_list = jieba_crafter.segment(text)
    assert len(crafted_chunk_list) == 14


def test_jieba_user_dir():
    jieba_segmenter = JiebaSegmenter()
    text = '今天是个大晴天！安迪回来以后，我们准备去动物园。thisisnotachineseword'
    crafted_chunk_list = jieba_segmenter.segment(text)

    assert len(crafted_chunk_list) == 15

    jieba_segmenter = JiebaSegmenter(user_dict_file=path_dict_file)
    text = '今天是个大晴天！安迪回来以后，我们准备去动物园。thisisnotachineseword'
    crafted_chunk_list = jieba_segmenter.segment(text)

    assert len(crafted_chunk_list) == 20


def test_jieba_user_dir_file_not_found():
    with pytest.raises(FileNotFoundError):
        JiebaSegmenter(user_dict_file='/this/path/does/not/exist.txt')
