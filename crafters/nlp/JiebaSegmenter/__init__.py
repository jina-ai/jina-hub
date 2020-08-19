from typing import Dict, List

from jina.executors.crafters import BaseSegmenter


class JiebaSegmenter(BaseSegmenter):
    """
    :class:`JiebaSegmenter` split the chinese text on the doc-level into words on the chunk-level with `jieba`.
    """

    def __init__(self, mode: str = 'accurate', *args, **kwargs):
        """

        :param mode: the jieba cut mode, accurate, all, search. default accurate
        """
        super().__init__(*args, **kwargs)
        if mode not in ('accurate', 'all', 'search'):
            raise ValueError('you must choose one of modes to cut the text: accurate, all, search.')
        self.mode = mode

    def craft(self, text: str, *args, **kwargs) -> List[Dict]:
        """
        Split the chinese text into words
        :param text: the raw text
        :return: a list of chunk dicts
        """
        import jieba
        if self.mode == 'search':
            words = jieba.cut_for_search(text)
        elif self.mode == 'all':
            words = jieba.cut(text, cut_all=True)
        else:
            words = jieba.cut(text)

        chunks = []
        for idx, word in enumerate(words):
            chunks.append(
                dict(text=word, offset=idx, weight=1.0))

        return chunks
