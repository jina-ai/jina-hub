from typing import Dict, List

from jina.executors.segmenters import BaseSegmenter


class JiebaSegmenter(BaseSegmenter):
    """
     :class:`JiebaSegmenter` split the chinese text
     on the doc-level into words on the chunk-level with `jieba`.
     :param mode: Jieba mode. The default mode is 'accurate'
     Options are:
     - accurate
     - search
     - all

    :type mode: str
    :raises:
        ValueError: If `mode` is not any of the expected modes
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments

    """

    def __init__(self, mode: str = 'accurate', *args, **kwargs):
        """Set Constructor."""
        super().__init__(*args, **kwargs)
        if mode not in ('accurate', 'all', 'search'):
            raise ValueError('you must choose one of modes to cut the text: accurate, all, search.')
        self.mode = mode

    def segment(self, text: str, *args, **kwargs) -> List[Dict]:
        """
        Split the chinese text into words.

        :param text: Raw text to be segmented
        :type text: str
        :param args:  Additional positional arguments
        :param kwargs: Additional keyword arguments
        :return: Sub-documents segmented
        :rtype: List[Dict]
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
