from .. import JiebaSegmenter


def test_jieba_crafter():
    jieba_crafter = JiebaSegmenter(mode='accurate')
    text = '今天是个大晴天！安迪回来以后，我们准备去动物园。'
    crafted_chunk_list = jieba_crafter.craft(text, 0)
    assert len(crafted_chunk_list) == 14
