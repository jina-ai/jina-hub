__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os

from jina.executors.crafters import BaseCrafter

TIKA_URL = 'http://0.0.0.0:9998'


class TikaExtractor(BaseCrafter):
    """
    :class:`TikaExtractor` Extracts text from files.
    """

    def __init__(self,
                 tika_ocr_strategy: str = 'ocr_only',
                 tika_extract_inline_images: str = 'true',
                 tika_ocr_language: str = 'eng',
                 tika_request_timeout: int = 600,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tika_ocr_strategy = tika_ocr_strategy
        self.tika_extract_inline_images = tika_extract_inline_images
        self.tika_ocr_language = tika_ocr_language
        self.tika_request_timeout = tika_request_timeout

    def post_init(self):
        super().post_init()
        self.start_tika()

    def start_tika(self):
        import requests
        from requests.exceptions import ConnectionError
        import time
        from subprocess import Popen, PIPE

        # Start tika server in subprocess
        tika_version = os.getenv('TIKA_VERSION')
        self.tika_process = Popen(
            ['java', '-jar', f'/tika-server-{tika_version}.jar', '-h', '0.0.0.0'],
            stdout=PIPE,
            stderr=PIPE)

        # Retry until tika server is started
        for _ in range(10):
            try:
                time.sleep(1)
                requests.get(TIKA_URL)
                return
            except ConnectionError:
                pass
        raise TimeoutError('Timeout when waiting for tika to start')

    def close(self):
        super().close()
        self.tika_process.kill()

    def craft(self, uri: str, buffer: bytes, *args, **kwargs):
        from tika import parser
        headers = {
            'X-Tika-PDFOcrStrategy': self.tika_ocr_strategy,
            'X-Tika-PDFextractInlineImages': str(self.tika_extract_inline_images),
            'X-Tika-OCRLanguage': self.tika_ocr_language
        }
        request_options = {
            'timeout': self.tika_request_timeout
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

        return dict(text=text)
