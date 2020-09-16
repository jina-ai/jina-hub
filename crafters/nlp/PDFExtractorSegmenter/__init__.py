from typing import Dict, List

from jina.executors.crafters import BaseSegmenter
import io


class PDFExtractorSegmenter(BaseSegmenter):
    """
    :class:`PDFExtractorSegmenter` Extracts data (text and images) from PDF.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def craft(self, uri: str, buffer: bytes, *args, **kwargs) -> List[Dict]:
        import fitz
        import PyPDF2
        chunks = []

        if uri:
            pdf_obj = open(uri, 'rb')
        elif buffer:
            pdf_obj = io.BytesIO(buffer)
        else:
            raise ValueError('No value found in "buffer" and "uri"')

        #Extract images
        pdf_doc = fitz.open(uri)
        for i in range(len(pdf_doc)):
            for img in pdf_doc.getPageImageList(i):
                xref = img[0]
                pix = fitz.Pixmap(pdf_doc, xref)
                if pix.n - pix.alpha < 4:   # if gray or RGB
                    pix.writePNG("p%s-%s.png" % (i, xref)) #Format is page, and image
                    chunks.append(
                        dict(blob=pix,  weight=1.0))
                else:                       # if CMYK:
                    pix1 = fitz.Pixmap(fitz.csRGB, pix) #Conver to RGB
                    pix1.writePNG("p%s-%s.png" % (i, xref))
                    chunks.append(
                        dict(blob=pix1, weight=1.0))
                    pix1 = None
                pix = None

        #Extract text
        text = ""
        pdf_reader = PyPDF2.PdfFileReader(pdf_obj)
        count = pdf_reader.numPages
        for i in range(count):
            page = pdf_reader.getPage(i)
            text += page.extractText()

        chunks.append(
            dict(text=text, weight=1.0))

        #Close pdf_obj
        pdf_obj.close()



        return chunks
