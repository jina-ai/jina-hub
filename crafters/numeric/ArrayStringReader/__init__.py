from typing import Dict

import numpy as np
from jina.executors.crafters import BaseCrafter


class ArrayStringReader(BaseCrafter):
    """
    :class:`ArrayStringReader` convertsthe string of numbers into a numpy array and save to the Document.
        Numbers are split on the provided delimiter, default is comma (,)
    """

    def __init__(self, delimiter: str = ',', as_type: str = 'float32', *args, **kwargs):
        """
        :param delimiter: delimiter between numbers
        :param as_type: type of number
        """
        super().__init__(*args, **kwargs)
        self.delimiter = delimiter
        self.as_type = as_type

    def craft(self, text: str, *args, **kwargs) -> Dict:
        """
        Split string into numbers and convert to numpy array

        :param text: the raw text
        :return: a dod dict with the numpy array
        """
        _string = text.split(self.delimiter)
        _array = np.array(_string)

        try:
            _array = _array.astype(self.as_type)
        except TypeError:
            self.logger.error(
                f'Data type {self.as_type} is not understood. '
                f'Please refer to the list of data types from Numpy.')
        except ValueError:
            self.logger.error(
                f'Data type mismatch. Cannot convert input to {self.as_type}.')

        return dict(weight=1., blob=_array)
