from collections import deque
from itertools import islice
from typing import Dict, List

from jina.executors.crafters import BaseSegmenter


class SlidingWindowSegmenter(BaseSegmenter):
    """
    :class:`SlidingWindowSegmenter` split the text on the doc-level into overlapping substrings on the chunk-level.
        The text is split into substrings of length ``window_size`` if possible.
        The degree of overlapping can be configured through the ``step_size`` parameter.
        The substrings that are shorter than the ``min_substring_len`` will be discarded.
    """

    def __init__(self,
                 window_size: int = 300,
                 step_size: int = 150,
                 min_substring_len: int = 1,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_size = window_size
        self.step_size = step_size
        self.min_substring_len = min_substring_len
        if self.min_substring_len > self.window_size:
            self.logger.warning(
                'the min_substring_len (={}) should be smaller to the window_size (={})'.format(
                    self.min_substring_len, self.window_size))
        if self.window_size <= 0:
            self.logger.warning(
                f'the window_size (={self.window_size}) should be larger than zero')
        if self.step_size > self.window_size:
            self.logger.warning(
                'the step_size (={}) should not be larger than the window_size (={})'.format(
                    self.window_size, self.step_size))

    def craft(self, text: str, *args, **kwargs) -> List[Dict]:
        """
        Split the text into overlapping chunks
        :param text: the raw text in string format
        :return: a list of chunk dicts
        """

        def sliding_window(iterable, size, step):
            i = iter(text)
            d = deque(islice(i, size),
                      maxlen=size)
            if not d:
                # empty text
                return results
            while True:
                yield iter(d)
                try:
                    d.append(next(i))
                except StopIteration:
                    return
                d.extend(next(i, None)
                         for _ in range(step - 1))

        chunks = [''.join(filter(None, list(chunk))) for chunk in
                  sliding_window(text, self.window_size, self.step_size)]
        results = []
        for idx, s in enumerate(chunks):
            if self.min_substring_len <= len(s):
                results.append(dict(
                    text=s,
                    offset=idx,
                    weight=1.0))
        return results
