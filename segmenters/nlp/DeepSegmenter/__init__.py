from typing import Dict, List

from jina.executors.segmenters import BaseSegmenter


class DeepSegmenter(BaseSegmenter):
    """
    Designed with ASR outputs in mind, DeepSegment uses BiLSTM +
    CRF for automatic sentence boundary detection.
    It outperforms the standard libraries (spacy, nltk, corenlp ..)
    on imperfect text, and performs similarly for perfectly punctuated text.

    Example: 'I am Batman i live in gotham'
            ->  # ['I am Batman', 'i live in gotham']

    Details: https://github.com/notAI-tech/deepsegment

    :param lang_code: en - english (Trained on data from various sources);
        fr - french (Only Tatoeba data); it - italian (Only Tatoeba data)
    :type lang_code: str
    :param checkpoint_name: Name to be used as checkpoint
    :type checkpoint_name: str
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    """
    def __init__(self, lang_code: str = 'en', checkpoint_name: str = None, *args, **kwargs):
        """Set constructor."""
        super().__init__(*args, **kwargs)
        self.lang_code = lang_code
        self.checkpoint_name = checkpoint_name

    def post_init(self):
        from deepsegment import DeepSegment
        self._segmenter = DeepSegment(self.lang_code, checkpoint_name=self.checkpoint_name)

    def segment(self, text: str, *args, **kwargs) -> List[Dict]:
        """
        Split the text into sentences.

        :param text: Raw text to be segmented
        :type text: str
        :param args:  Additional positional arguments
        :param kwargs: Additional keyword arguments
        :return: List of sub-docuemnt dicts with the cropped images
        :rtype: List[Dict]
        """

        results = []
        for idx, s in enumerate(self._segmenter.segment_long(text)):
            results.append(dict(
                text=s,
                offset=idx,
                weight=1.0))
        return results
