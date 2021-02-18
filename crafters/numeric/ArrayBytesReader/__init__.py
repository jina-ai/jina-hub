from typing import Dict

import numpy as np
from jina.executors.crafters import BaseCrafter


class ArrayBytesReader(BaseCrafter):
    """
    Convert a byte stream into a numpy array and save to the Document.

    The size of the vectors is provided in the constructor
    so that the numpy array can be interpreted properly.

    :param as_type: The numpy array will be this type
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    """

    def __init__(self, as_type: str = 'float32', *args, **kwargs):
        """Set constructor"""
        super().__init__(*args, **kwargs)
        self.as_type = as_type

    def craft(self, buffer: bytes, *args, **kwargs) -> Dict:
        """
        Split string into numbers and convert to numpy array.

        :param buffer: the bytes representing the array
        :param args:  Additional positional arguments
        :param kwargs: Additional keyword arguments
        :return: a chunk dict with the numpy array
        """
        _array = np.frombuffer(buffer, self.as_type)
        return dict(blob=_array)
