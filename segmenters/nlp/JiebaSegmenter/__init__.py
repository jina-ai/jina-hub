from typing import Dict, List

from jina.executors.decorators import single
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
    :param user_dict_file: Path to a custom dictionary.
        This custom dictionary extends the default dictionary.
        `See here for the format of the dict https://github.com/fxsjy/jieba#load-dictionary`_
    :type user_dict_file: str
    :raises:
        ValueError: If `mode` is not any of the expected modes
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments

    """

    def __init__(self, mode: str = 'accurate', user_dict_file: str = None, *args, **kwargs):
        """Set Constructor."""
        super().__init__(*args, **kwargs)
        if mode not in ('accurate', 'all', 'search'):
            raise ValueError('you must choose one of modes to cut the text: accurate, all, search.')
        self.mode = mode
        self.user_dict_file = user_dict_file

    def post_init(self):
        """ Load custom dict if provided. Raise FileNotFoundError if dict does not exist."""
        super().post_init()

        if self.user_dict_file is not None:
            import os
            import jieba
            if not os.path.exists(self.user_dict_file):
                raise FileNotFoundError(f'User dictionary can not be found at {self.user_dict_file}')
            jieba.load_userdict(self.user_dict_file)

    @single
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
