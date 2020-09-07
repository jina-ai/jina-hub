__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from jina.executors.crafters import BaseCrafter


class PDFTextExtractor(BaseCrafter):
    """
    :class:`PDFTextExtractor` Extracts text from PDF.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def craft(self, uri: str, buffer: bytes, *args, **kwargs):
        import PyPDF2

        text = ""
        if buffer:
            pdf_obj = buffer
        elif uri:
            pdf_obj = open(uri, 'rb')
        else:
            raise ValueError('no value found in "buffer" and "uri"')

        pdf_reader = PyPDF2.PdfFileReader(pdf_obj)
        count = pdf_reader.numPages
        for i in range(count):
            page = pdf_reader.getPage(i)
            text += page.extractText()

        pdf_obj.close()

        return text
