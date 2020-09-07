__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from jina.executors.crafters import BaseCrafter


class PDFTextExtractor(BaseCrafter):
    import PyPDF2
    """
    :class:`PDFTextExtractor` Extracts text from PDF.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # your customized __init__ below
        raise NotImplementedError

    def craft(self, *args, **kwargs):
        raise NotImplementedError
