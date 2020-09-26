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

    def start_tika(self):
        from subprocess import Popen
        tika_version = os.getenv('TIKA_VERSION')
        _ = Popen(
            ['java', '-jar', f'/tika-server-{tika_version}.jar', '-h', '0.0.0.0'],
            stdout=open('stdout.log', 'w'),
            stderr=open('stderr.log', 'w'))

        response = None
        while not response:
            try:
                response = requests.get(TIKA_URL)
            except ConnectionError:
                pass

    def __init__(self, *args, **kwargs):
        self.start_tika()
        super().__init__(*args, **kwargs)

    def craft(self, uri: str, *args, **kwargs):
        from tika import parser
        headers = {
            'X-Tika-PDFOcrStrategy': os.getenv('TIKA_OCR_STRATEGY', 'ocr_only'),
            'X-Tika-PDFextractInlineImages': os.getenv('TIKA_EXTRACT_INLINE_IMAGES', 'true'),
            'X-Tika-OCRLanguage': os.getenv('TIKA_OCR_LANGUAGE', 'eng')
        }
        request_options = {
            'timeout': int(os.getenv('TIKA_TIMEOUT', '600'))
        }

        extract = {}
        result = parser.from_file(filename=uri,
                                  serverEndpoint=TIKA_URL,
                                  service='all',
                                  xmlContent=False,
                                  headers=headers,
                                  requestOptions=request_options)
        if 'status' in result:
            if result['status'] == 200:
                extract['text'] = result['content']
                extract['metadata'] = result['metadata']

        return extract
