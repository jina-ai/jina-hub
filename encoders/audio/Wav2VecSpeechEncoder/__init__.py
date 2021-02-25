__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
from typing import Optional

import numpy as np

from jina.executors.decorators import batching, as_ndarray
from jina.executors.encoders import BaseAudioEncoder
from jina.executors.encoders.frameworks import BaseTorchEncoder
from jina.excepts import PretrainedModelFileDoesNotExist
from jina.helper import cached_property


class Wav2VecSpeechEncoder(BaseTorchEncoder, BaseAudioEncoder):
    """
    Use a pre-trained model (`wav2vec`) to encode audio signal.

    :class:`Wav2VecSpeechEncoder` is a speech encoder based on `wav2vec`,
        an unsupervised pre-trained model for speech recognition presented and implemented
        by Facebook: https://github.com/pytorch/fairseq/tree/master/examples/wav2vec
        It uses a pre-trained model to encode an audio signal from
        a `Batch x Signal Length` ndarray into a `Batch x Concatenated Features` ndarray,
        and produces a representation for each time step at a rate of 100 Hz.

    :param model_path: the path of the pre-trained model.
        The pre-trained model can be downloaded at
        https://github.com/pytorch/fairseq/tree/master/examples/wav2vec/README.md#wav2vec
    :param input_sample_rate: input sampling rate in Hz (22050 by default)
    """

    def __init__(self,
                 model_path: Optional[str] = '/tmp/wav2vec_large.pt',
                 input_sample_rate: int = 22050,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.model_path = model_path
        self.input_sample_rate = input_sample_rate

    def post_init(self):
        """Load Wav2Vec model"""
        super().post_init()
        if self.model_path and os.path.exists(self.model_path):
            import torch
            from fairseq.models.wav2vec import Wav2VecModel
            cp = torch.load(self.model_path, map_location=torch.device('cpu'))
            self.model = Wav2VecModel.build_model(cp['args'], task=None)
            self.model.load_state_dict(cp['model'])
            self.model.eval()
            self.to_device(self.model)
            self._tensor_func = torch.tensor
        else:
            raise PretrainedModelFileDoesNotExist(f'model at {self.model_path} does not exist')

    @batching
    @as_ndarray
    def encode(self, data: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Resample  input audio signal to 16kHz.

        Segments the resampled signal of each Doc into `wav2vec` frames,
        encodes the frames and concatenates Doc frame embeddings into a
        single Doc embedding.

        :param data: A`Batch x Signal Length` ndarray, where
            `Signal Length` is a number of samples
        :return: A `Batch x Concatenated Features` ndarray,
            where `Concatenated Features` is a 512-dimensional feature
            vector times the number of the wav2vec frames.
        """
        assert data.shape[1] >= 465, 'the signal must have at least 465 samples'
        from librosa import resample
        embeds = []
        with self.session():
            for chunk_data in data:
                resampled_signal = resample(chunk_data, self.input_sample_rate, 16000)
                signal_tensor = self.array2tensor(resampled_signal.reshape(1, -1))
                features = self.model.feature_extractor(signal_tensor)
                embed_tensor = self.model.feature_aggregator(features)[0]
                chunk_embed = self.tensor2array(embed_tensor).T.flatten()
                embeds.append(chunk_embed)
        return embeds

    def array2tensor(self, array):
        """Transform array into tensor"""
        tensor = self._tensor_func(array)
        return tensor.cuda() if self.on_gpu else tensor

    def tensor2array(self, tensor):
        """Transform tensor into array"""
        return tensor.cuda().numpy() if self.on_gpu else tensor.numpy()

    @cached_property
    def session(self):
        return self.get_session()

    def get_session(self):
        """Get no_grad from torch"""
        from torch import no_grad
        return no_grad