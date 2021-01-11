import io
from typing import Dict, List

import numpy as np

from jina.executors.segmenters import BaseSegmenter


class PDFExtractorSegmenter(BaseSegmenter):
    """
    :class:`PDFExtractorSegmenter` Extracts data (text and images) from PDF.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def segment(self, uri: str, buffer: bytes, *args, **kwargs) -> List[Dict]:
        import fitz
        import PyPDF2

        if uri:
            pdf_img = fitz.open(uri)
            pdf_text = open(uri, 'rb')
        elif buffer:
            pdf_img = fitz.open(stream=buffer, filetype='pdf')
            pdf_text = io.BytesIO(buffer)
        else:
            raise ValueError('No value found in "buffer" or "uri"')

        chunks = []
        # Extract images
        with pdf_img:
            for page in range(len(pdf_img)):
                for img in pdf_img.getPageImageList(page):
                    xref = img[0]
                    pix = fitz.Pixmap(pdf_img, xref)
                    np_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n).astype('float32')
                    if pix.n - pix.alpha < 4:  # if gray or RGB
                        chunks.append(
                            dict(blob=np_arr, weight=1.0, mime_type='image/png'))
                    else:  # if CMYK:
                        pix = fitz.Pixmap(fitz.csRGB, pix)  # Convert to RGB
                        np_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n).astype(
                            'float32')
                        chunks.append(
                            dict(blob=np_arr, weight=1.0, mime_type='image/png'))

        # Extract text
        with pdf_text:
            text = ""
            pdf_reader = PyPDF2.PdfFileReader(pdf_text)
            count = pdf_reader.numPages
            for page in range(count):
                page = pdf_reader.getPage(page)
                text += page.extractText()
            if text:
                chunks.append(
                    dict(text=text, weight=1.0, mime_type='text/plain'))

        return chunks
