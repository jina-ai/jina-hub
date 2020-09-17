from typing import Dict, List

from jina.executors.crafters import BaseSegmenter
import io
import numpy as np



class PDFExtractorSegmenter(BaseSegmenter):
    """
    :class:`PDFExtractorSegmenter` Extracts data (text and images) from PDF.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def craft(self, uri: str, buffer: bytes, *args, **kwargs) -> List[Dict]:
        import fitz
        import PyPDF2

        if uri:
            pdf_img = fitz.open(uri)
            pdf_text = open(uri, 'rb')
        elif buffer:
            pdf_text = io.BytesIO(buffer)
            pdf_img = fitz.open(stream=buffer, filetype="pdf")
        else:
            raise ValueError('No value found in "buffer" and "uri"')

        chunks = []
        with pdf_img:
            #Extract images
            for i in range(len(pdf_img)):
                for img in pdf_img.getPageImageList(i):
                    xref = img[0]
                    pix = fitz.Pixmap(pdf_img, xref)
                    np_arr = pix2np(pix)
                    if pix.n - pix.alpha < 4:   # if gray or RGB
                        chunks.append(
                            dict(blob=np_arr,  weight=1.0))
                    else:                       # if CMYK:
                        pix1 = fitz.Pixmap(fitz.csRGB, pix) #Conver to RGB
                        np_arr1 = pix2np(pix1)
                        chunks.append(
                            dict(blob=np_arr1, weight=1.0))

        #Extract text
        with pdf_text:
            text = ""
            pdf_reader = PyPDF2.PdfFileReader(pdf_text)
            count = pdf_reader.numPages
            for i in range(count):
                page = pdf_reader.getPage(i)
                text += page.extractText()

            if text:
                chunks.append(
                    dict(text=text, weight=1.0))

        return chunks

def pix2np(pix):
    im = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    im = np.ascontiguousarray(im[..., [2, 1, 0]])  # rgb to bgr
    return im