from typing import Dict

import numpy as np
from jina.executors.crafters import BaseCrafter


class ArrayBytesReader(BaseCrafter):
    """
    :class:`ArrayBytesReader` converts a byte stream into a numpy array and save to the Document.
        The size of the vectors is provided in the constructor so that the numpy array can be interpreted properly
    """

    def __init__(self, as_type: str = 'float32', *args, **kwargs):
        """
        :param as_type: type of number
        """
        super().__init__(*args, **kwargs)
        self.as_type = as_type

    def craft(self, buffer: bytes, *args, **kwargs) -> Dict:
        """
        Split string into numbers and convert to numpy array

        :param buffer: the bytes representing the array
        :return: a chunk dict with the numpy array
        """
        _array = np.frombuffer(buffer, self.as_type)
        return dict(weight=1., blob=_array)
