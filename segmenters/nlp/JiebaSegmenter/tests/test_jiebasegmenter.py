import os

import pytest

from .. import JiebaSegmenter

cur_dir = os.path.dirname(os.path.abspath(__file__))
path_dict_file = os.path.join(cur_dir, 'dict.txt')

def test_jieba_crafter():
    jieba_crafter = JiebaSegmenter(mode='accurate')
    text = '今天是个大晴天！安迪回来以后，我们准备去动物园。'
    crafted_chunk_list = jieba_crafter.segment(text, 0)
    assert len(crafted_chunk_list) == 14


def test_jieba_multi_processing_working():
    import jieba

    jieba_crafter = JiebaSegmenter(mode='accurate', pool_size=2)
    text = '今天是个大晴天！安迪回来以后，我们准备去动物园。'
    crafted_chunk_list = jieba_crafter.segment(text, 0)

    assert jieba.pool is not None
    assert len(crafted_chunk_list) == 14


@pytest.mark.parametrize("pool_size, pool_is_none", [(-1, True), (0, True), (1, True), (2, False), (None, False)])
def test_jieba_multi_processing_setup(pool_size, pool_is_none):
    import jieba

    JiebaSegmenter(pool_size=pool_size)
    if pool_is_none:
        assert jieba.pool is None
    else:
        assert jieba.pool is not None


def test_jieba_user_dir():
    jieba_segmenter = JiebaSegmenter()
    text = '今天是个大晴天！安迪回来以后，我们准备去动物园。thisisnotachineseword'
    crafted_chunk_list = jieba_segmenter.segment(text, 0)

    assert len(crafted_chunk_list) == 15

    jieba_segmenter = JiebaSegmenter(user_dict_file=path_dict_file)
    text = '今天是个大晴天！安迪回来以后，我们准备去动物园。thisisnotachineseword'
    crafted_chunk_list = jieba_segmenter.segment(text, 0)

    assert len(crafted_chunk_list) == 20


def test_jieba_user_dir_file_not_found():
    with pytest.raises(FileNotFoundError):
        JiebaSegmenter(user_dict_file='/this/path/does/not/exist.txt')
