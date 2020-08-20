import os
import pytest
import numpy as np

from jina.executors import BaseExecutor
from tests.unit.executors import ExecutorTestCase


class ImageTestCase(ExecutorTestCase):
    @property
    def workspace(self):
        return os.path.join(os.environ['TEST_WORKDIR'], 'test_tmp')

    @property
    def target_output_dim(self):
        return self._target_output_dim

    @target_output_dim.setter
    def target_output_dim(self, output_dim):
        self._target_output_dim = output_dim

    @property
    def input_dim(self):
        return self._input_dim

    @input_dim.setter
    def input_dim(self, input_dim):
        self._input_dim = input_dim

    def get_encoder(self):
        encoder = self._get_encoder(self.metas)
        if encoder is not None:
            encoder.workspace = self.workspace
            self.add_tmpfile(encoder.workspace)
        return encoder

    def _get_encoder(self, metas):
        return None