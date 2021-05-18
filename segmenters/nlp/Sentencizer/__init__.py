import re
from typing import Dict, List, Optional

from jina import requests, Document, Executor


class Sentencizer(Executor):
    """
    :class:`Sentencizer` split the text on the doc-level
    into sentences on the chunk-level with a rule-base strategy.
    The text is split by the punctuation characters listed in ``punct_chars``.
    The sentences that are shorter than the ``min_sent_len``
    or longer than the ``max_sent_len`` after stripping will be discarded.

    :param min_sent_len: the minimal number of characters,
        (including white spaces) of the sentence, by default 1.
    :param max_sent_len: the maximal number of characters,
        (including white spaces) of the sentence, by default 512.
    :param punct_chars: the punctuation characters to split on,
        whatever is in the list will be used,
        for example ['!', '.', '?'] will use '!', '.' and '?'
    :param uniform_weight: the definition of it should have
        uniform weight or should be calculated
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments

    """

    def __init__(self,
                 min_sent_len: int = 1,
                 max_sent_len: int = 512,
                 punct_chars: Optional[List[str]] = None,
                 uniform_weight: bool = True,
                 *args, **kwargs):
        """Set constructor."""
        super().__init__(*args, **kwargs)
        self.min_sent_len = min_sent_len
        self.max_sent_len = max_sent_len
        self.punct_chars = punct_chars
        self.uniform_weight = uniform_weight
        if not punct_chars:
            self.punct_chars = ['!', '.', '?', '։', '؟', '۔', '܀', '܁', '܂', '‼', '‽', '⁇', '⁈', '⁉', '⸮', '﹖', '﹗',
                                '！', '．', '？', '｡', '。', '\n']
        if self.min_sent_len > self.max_sent_len:
            self.logger.warning('the min_sent_len (={}) should be smaller or equal to the max_sent_len (={})'.format(
                self.min_sent_len, self.max_sent_len))
        self._slit_pat = re.compile('\s*([^{0}]+)(?<!\s)[{0}]*'.format(''.join(set(self.punct_chars))))

    @requests
    def segment(self, docs: 'DocumentArray', **kwargs):

        for doc in docs:
            chunks = self._segment_one(doc.text)
            for chunk in chunks:
                if not chunk.mime_type:
                    chunk.mime_type = doc.mime_type
            doc.chunks.extend(chunks)

        return docs

    def _segment_one(self, text: str):
        """
        Split the text into sentences.

        :param text: the text to segment
        :return: a list of chunks with the split sentences

        """
        chunks = []
        splits = [(m.group(0), m.start(), m.end()) for m in
               re.finditer(self._slit_pat, text)]
        if not splits:
            splits = [(text, 0, len(text))]
        for ci, (r, start, end) in enumerate(splits):
            chunk_text = re.sub('\n+', ' ', r).strip()
            chunk_text = chunk_text[:self.max_sent_len]
            if len(chunk_text) > self.min_sent_len:
                chunks.append(Document(
                    text=chunk_text,
                    offset=ci,
                    weight=1.0 if self.uniform_weight else len(chunk_text) / len(text),
                    location=[start, end]
                ))
        return chunks
