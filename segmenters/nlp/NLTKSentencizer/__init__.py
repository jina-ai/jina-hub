from typing import Dict, List, Optional
from jina.executors.segmenters import BaseSegmenter
from jina.executors.decorators import single


class NLTKSentencizer(BaseSegmenter):
    """
    Segment text into sentences using :class:`nltk.PunktSentenceTokenizer`
    for the specified language.

    Example:

    >>> sentencizer = NLTKSentencizer() # doctest:+ELLIPSIS
    NLTKSentencizer@4...[W]:__main__.NLTKSentencizer works on keys, but no keys are specified
    NLTKSentencizer@...[I]:post_init may take some time...
    NLTKSentencizer@...[I]:NLTK english sentence tokenizer ready.
    NLTKSentencizer@...[I]:post_init may take some time takes ... seconds (...s)
    >>> text = "Today is a good day. Can't wait for tomorrow!"
    >>> sentencizer.segment([text])
    [{'text': 'Today is a good day.', 'offset': 0, 'location': [0, 20]}, {'text': "Can't wait for tomorrow!", 'offset': 1, 'location': [21, 45]}]


    :param language: default='english'. Lowercased language name to initialize the sentence tokenizer, accepted languages are listed in :attr:`SUPPORTED_LANGUAGES`.
    :type language: Optional[str]
    :param resource_url: default=None. Optional parameter to specify the location for tokenizer model, overrides `language`. For a file path, should follow `file://path_to_resource` format. For more information, see https://www.nltk.org/data.html
    :type resource_url: Optional[str]
    :param args: Additional positional arguments
    :param kwargs: Additional keyword arguments
    """

    SUPPORTED_LANGUAGES = [
        'czech',
        'danish',
        'dutch',
        'english',
        'estonian',
        'finnish',
        'french',
        'german',
        'greek',
        'italian',
        'norwegian',
        'polish',
        'portuguese',
        'russian',
        'slovene',
        'spanish',
        'swedish',
        'turkish',
    ]

    def __init__(
        self,
        language: Optional[str] = 'english',
        resource_url: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """Set constructor"""
        super().__init__(*args, **kwargs)
        if resource_url:
            self.resource_url = resource_url
            self.language = None
        else:
            self.resource_url = f'tokenizers/punkt/{language}.pickle'
            self.language = language

    def post_init(self):
        from nltk.data import load

        try:
            self._sent_tokenizer = load(self.resource_url)
        except LookupError:
            if self.language in self.SUPPORTED_LANGUAGES:
                try:
                    import nltk

                    nltk.download('punkt')
                    self._sent_tokenizer = load(self.resource_url)
                except:
                    self.logger.error(
                        f'Please ensure that "nltk_data" folder is accessible to your working directory.'
                    )
                    raise
            elif self.language:
                self.logger.error(
                    f'The language you specified ("{self.language}") is not supplied by NLTK.'
                    f'Please ensure that language is one of the acceptable languages listed in '
                    f'{self.SUPPORTED_LANGUAGES}.\nFor latest list of supported languages check out '
                    f'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml with '
                    f'id="punkt"'
                )
                raise
            else:
                self.logger.error(
                    f'The resource_url you provided could not be loaded: {self.resource_url}'
                )
                raise
        self.logger.info(f'NLTK {self.language} sentence tokenizer ready.')

    @single
    def segment(self, text: str, *args, **kwargs) -> List[Dict]:
        """
        Segment text into sentences.

        :param text: The text to be sentencized.
        :type text: str
        :param args: Additional positional arguments
        :param kwargs: Additional keyword arguments
        :return: List of dictonaries representing sentences with three keys: ``text``, for representing the text of the sentence; ``offset``, representing the order of the sentence within input text; ``location``, a list with start and end indeces for the sentence.
        :rtype: List[Dict]
        """
        sentences = self._sent_tokenizer.tokenize(text)
        results = []
        start = 0
        for i, s in enumerate(sentences):
            start = text[start:].find(s) + start
            end = start + len(s)
            results.append({'text': s, 'offset': i, 'location': [start, end]})
            start = end

        return results
