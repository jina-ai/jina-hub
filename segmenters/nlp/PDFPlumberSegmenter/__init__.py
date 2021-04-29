import io
from typing import Dict, List

import fitz
import numpy as np
import pdfplumber
from jina.executors.decorators import single
from jina.executors.segmenters import BaseSegmenter
from pdftitle import get_title_from_file, get_title_from_io


class PDFPlumberSegmenter(BaseSegmenter):
    """
    :class:`PDFPlumberSegmenter` Extracts data (text and images) from PDF files.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _extract_img(self, pdf_img, chunks):
        with pdf_img:
            for page in range(len(pdf_img)):
                for img in pdf_img.getPageImageList(page):
                    xref = img[0]
                    pix = fitz.Pixmap(pdf_img, xref)
                    # read data from buffer and reshape the array into 3-d format
                    np_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n).astype('float32')
                    if pix.n - pix.alpha < 4:  # if gray or RGB
                        if pix.n == 1:  # convert gray to rgb
                            np_arr_rgb = np.concatenate((np_arr,) * 3, -1)
                            chunks.append(dict(blob=np_arr_rgb, weight=1.0, mime_type='image/png'))
                        elif pix.n == 4:  # remove transparency layer
                            np_arr_rgb = np_arr[..., :3]
                            chunks.append(dict(blob=np_arr_rgb, weight=1.0, mime_type='image/png'))
                        else:
                            chunks.append(dict(blob=np_arr, weight=1.0, mime_type='image/png'))
                    else:  # if CMYK:
                        pix = fitz.Pixmap(fitz.csRGB, pix)  # Convert to RGB
                        np_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n).astype(
                            'float32')
                        chunks.append(
                            dict(blob=np_arr, weight=1.0, mime_type='image/png'))

    def _extract_text(self, pdf_content, chunks, title):
        with pdf_content:
            count = len(pdf_content.pages)
            for i in range(count):
                page = pdf_content.pages[i]
                text_page = page.extract_text(x_tolerance=1, y_tolerance=1)
                text_page = text_page.replace(u'\xa0', u'')
                if title:
                    chunks.append(dict(text=title, weight=1.0, mime_type='text/plain', tags={'title': True}))
                if text_page:
                    chunks.append(dict(text=text_page, weight=1.0, mime_type='text/plain'))

    @single(slice_nargs=3)
    def segment(self, uri: str, buffer: bytes, mime_type: str, *args, **kwargs) -> List[Dict]:
        """
        Segements PDF files. Extracts data from them.

        Checks if the input is a string of the filename,
        or if it's the file in bytes.
        It will then extract the data from the file, creating a list for images,
        and text.

        :param uri: File name of PDF
        :type uri: str
        :param buffer: PDF file in bytes
        :type buffer: bytes
        :param mime_type: the type of data
        :returns: A list of documents with the extracted data
        :rtype: List[Dict]
        """
        chunks = []
        if mime_type != 'application/pdf':
            return chunks

        if uri:
            try:
                pdf_img = fitz.open(str(uri))
                pdf_content = pdfplumber.open(uri)
                title = get_title_from_file(uri)
            except Exception as ex:
                self.logger.error(f'Failed to open {uri}: {ex}')
                return chunks
        elif buffer:
            try:
                pdf_img = fitz.open(stream=buffer, filetype='pdf')
                pdf_content = pdfplumber.open(io.BytesIO(buffer), password=b"")
                title = get_title_from_io(io.BytesIO(buffer))
            except Exception as ex:
                self.logger.error(f'Failed to load from buffer')
                return chunks
        else:
            self.logger.warning('No value found in `buffer` or `uri`')
            return chunks
        # Extract images
        self._extract_img(pdf_img, chunks)
        # Extract text
        self._extract_text(pdf_content, chunks, title)
        return chunks
