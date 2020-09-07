__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from jina.executors.crafters import BaseCrafter


class PDFTextExtractor(BaseCrafter):

    """
    :class:`PDFTextExtractor` Extracts text from PDF.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def craft(self, uri: str, *args, **kwargs):
        import PyPDF2
        pdf_obj = open(uri, 'rb')
        pdf_reader = PyPDF2.PdfFileReader(pdf_obj)
        page_obj = pdf_reader.getPage(0)
        text = page_obj.extractText()
        return text

