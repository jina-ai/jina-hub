__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os

import requests
from requests.exceptions import ConnectionError
from jina.executors.crafters import BaseCrafter

TIKA_URL = 'http://0.0.0.0:9998'


class TikaExtractor(BaseCrafter):
    """
    :class:`TikaExtractor` Extracts text from files.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def post_init(self):
        super().post_init()
        self.start_tika()

    def start_tika(self):
        from subprocess import Popen
        tika_version = os.getenv('TIKA_VERSION')
        self.tika_process = Popen(
            ['java', '-jar', f'/tika-server-{tika_version}.jar', '-h', '0.0.0.0'],
            stdout=open('stdout.log', 'w'),
            stderr=open('stderr.log', 'w'))

        response = None
        while not response:
            try:
                response = requests.get(TIKA_URL)
            except ConnectionError:
                pass

    def close(self):
        self.tika_process.kill()

    def craft(self, uri: str, buffer: bytes, *args, **kwargs):
        from tika import parser
        headers = {
            'X-Tika-PDFOcrStrategy': os.getenv('TIKA_OCR_STRATEGY', 'ocr_only'),
            'X-Tika-PDFextractInlineImages': os.getenv('TIKA_EXTRACT_INLINE_IMAGES', 'true'),
            'X-Tika-OCRLanguage': os.getenv('TIKA_OCR_LANGUAGE', 'eng')
        }
        request_options = {
            'timeout': int(os.getenv('TIKA_TIMEOUT', '600'))
        }

        text = ""
        if buffer:
            result = parser.from_buffer(string=buffer,
                                        serverEndpoint=TIKA_URL,
                                        xmlContent=False,
                                        headers=headers,
                                        requestOptions=request_options)
        elif uri:
            result = parser.from_file(filename=uri,
                                      serverEndpoint=TIKA_URL,
                                      service='all',
                                      xmlContent=False,
                                      headers=headers,
                                      requestOptions=request_options)
        else:
            raise ValueError('No value found in "buffer" and "uri"')

        if 'status' in result:
            if result['status'] == 200:
                text = result['content']

        return text
