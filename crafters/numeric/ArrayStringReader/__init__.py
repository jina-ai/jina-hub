__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Dict

import numpy as np

from jina.executors.decorators import single
from jina.executors.crafters import BaseCrafter


class ArrayStringReader(BaseCrafter):
    """
    Convert string of numbers into a numpy array and save to the Document.

    Numbers are split on the provided delimiter, default is comma (,)

    :param delimiter: delimiter between numbers
    :param as_type: The numpy array will be this type
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    """

    def __init__(self, delimiter: str = ',', as_type: str = 'float32', *args, **kwargs):
        """Set constructor."""
        super().__init__(*args, **kwargs)
        self.delimiter = delimiter
        self.as_type = as_type

    @single
    def craft(self, text: str, *args, **kwargs) -> Dict:
        """
        Split string into numbers and convert to numpy array.

        :param text: the raw text to be converted
        :return: a dict with the created numpy array
        :param args:  Additional positional arguments
        :param kwargs: Additional keyword arguments
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

        return dict(blob=_array)
