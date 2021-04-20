__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import time

from typing import Optional, Dict

import numpy as np

from jina.executors.decorators import batching, as_ndarray
from jina.executors.devices import TorchDevice
from jina.executors.encoders import BaseEncoder

if False:
    import torch

HTTP_SERVICE_UNAVAILABLE = 503


class TransformerTorchEncoder(TorchDevice, BaseEncoder):
    """
    Wraps the pytorch version of transformers from huggingface.

    :param pretrained_model_name_or_path: Either:
        - a string, the `model id` of a pretrained model hosted
            inside a model repo on huggingface.co, e.g.: ``bert-base-uncased``.
        - a path to a `directory` containing model weights saved using
            :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.:
            ``./my_model_directory/``.
    :param base_tokenizer_model: The name of the base model to use for creating
        the tokenizer. If None, will be equal to `pretrained_model_name_or_path`.
    :param pooling_strategy: the strategy to merge the word embeddings into the
        chunk embedding. Supported strategies include 'cls', 'mean', 'max', 'min'.
    :param layer_index: index of the transformer layer that is used to create
        encodings. Layer 0 corresponds to the embeddings layer
    :param max_length: the max length to truncate the tokenized sequences to.
    :param acceleration: The method to accelerate encoding. The available options are:
        - ``'amp'``, which uses `automatic mixed precision
            `<https://pytorch.org/docs/stable/amp.html>`_ autocasting.
            This option is only available on GPUs that support it
            (architecture newer than or equal to NVIDIA Volatire).
        - ``'quant'``, which uses dynamic quantization on the transformer model.
            See `this tutorial
            <https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html>`_
            for more information. This option is currently not supported on GPUs.
    :param embedding_fn_name: name of the function to be called from the `model` to do the embedding. `__call__` by default.
            Other possible values would `embed_questions` for `RetriBert` based models
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments

    ..note::
        While acceleration methods can significantly speed up the encoding,
        they result in loss of precision. Make sure that the tradeoff is
        worthwhile for your use case.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = 'sentence-transformers/distilbert-base-nli-stsb-mean-tokens',
        base_tokenizer_model: Optional[str] = None,
        pooling_strategy: str = 'mean',
        layer_index: int = -1,
        max_length: Optional[int] = None,
        acceleration: Optional[str] = None,
        embedding_fn_name: str = '__call__',
        api_token: Optional[str] = None,
        max_retries: int = 20,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.base_tokenizer_model = (
            base_tokenizer_model or pretrained_model_name_or_path
        )
        self.pooling_strategy = pooling_strategy
        self.layer_index = layer_index
        self.max_length = max_length
        self.acceleration = acceleration
        self.embedding_fn_name = embedding_fn_name
        self.api_token = api_token
        self.max_retries = max_retries

        if self.pooling_strategy == 'auto':
            self.pooling_strategy = 'cls'
            self.logger.warning(
                '"auto" pooling_strategy is deprecated, Defaulting to '
                ' "cls" to maintain the old default behavior.'
            )

        if self.pooling_strategy not in ['cls', 'mean', 'max', 'min']:
            self.logger.error(
                f'pooling strategy not found: {self.pooling_strategy}.'
                ' The allowed pooling strategies are "cls", "mean", "max", "min".'
            )
            raise NotImplementedError

        if self.acceleration not in [None, 'amp', 'quant']:
            self.logger.error(
                f'acceleration not found: {self.acceleration}.'
                ' The allowed accelerations are "amp" and "quant".'
            )
            raise NotImplementedError

        if self.api_token is not None and self.layer_index != -1:
            self.logger.error(
                f'layer_index {self.layer_index} not available via the huggingface API.'
                ' Please set the layer_index to -1 or disable huggingface API usage.'
            )
            raise ValueError

    def post_init(self):
        """Load the transformer model and encoder"""
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_tokenizer_model)

        if self.api_token is None:
            self.model = AutoModel.from_pretrained(
                self.pretrained_model_name_or_path, output_hidden_states=True
            )
            self.to_device(self.model)

            if self.acceleration == 'quant' and not self.on_gpu:
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
        else:
            self._api_call('Scotty, please warmup.')

    def amp_accelerate(self):
        """Check acceleration method """
        import torch
        from contextlib import nullcontext

        if self.acceleration == 'amp':
            return torch.cuda.amp.autocast()
        else:
            return nullcontext()

    def _api_call(self, query):
        import requests

        retries = 0
        headers = {"Authorization": f"Bearer {self.api_token}"}
        api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.pretrained_model_name_or_path}"

        while retries < self.max_retries:
            retries += 1
            r = requests.post(api_url, json=query, headers=headers)
            if r.status_code == HTTP_SERVICE_UNAVAILABLE:
                self.logger.info('Service is currently unavailable. Model is loading.')
                time.sleep(3)
            else:
                break
        return r.json()

    def _compute_embedding(self, hidden_states: 'torch.Tensor', input_tokens: Dict):
        import torch

        n_layers = len(hidden_states)
        if self.layer_index not in list(range(-n_layers, n_layers)):
            self.logger.error(
                f'Invalid value {self.layer_index} for `layer_index`,'
                f' for the model {self.pretrained_model_name_or_path}'
                f' valid values are integers from {-n_layers} to {n_layers - 1}.'
            )
            raise ValueError

        if self.pooling_strategy == 'cls' and not self.tokenizer.cls_token:
            self.logger.error(
                f'You have set pooling_strategy to "cls", but the tokenizer'
                f' for the model {self.pretrained_model_name_or_path}'
                f' does not have a cls token set.'
            )
            raise ValueError

        fill_vals = {'cls': 0.0, 'mean': 0.0, 'max': -np.inf, 'min': np.inf}
        fill_val = torch.tensor(fill_vals[self.pooling_strategy], device=self.device)

        layer = hidden_states[self.layer_index]
        attn_mask = input_tokens['attention_mask'].unsqueeze(-1).expand_as(layer)
        layer = torch.where(attn_mask.bool(), layer, fill_val)

        if self.pooling_strategy == 'cls':
            CLS = self.tokenizer.cls_token_id
            ind = torch.nonzero(input_tokens['input_ids'] == CLS)[:, 1]
            ind = ind.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, layer.shape[2])
            embeddings = torch.gather(layer, 1, ind).squeeze(dim=1)
        elif self.pooling_strategy == 'mean':
            embeddings = layer.sum(dim=1) / attn_mask.sum(dim=1)
        elif self.pooling_strategy == 'max':
            embeddings = layer.max(dim=1).values
        elif self.pooling_strategy == 'min':
            embeddings = layer.min(dim=1).values

        return embeddings.cpu().numpy()

    @batching
    @as_ndarray
    def encode(self, content: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':
        """
        Encode an array of string in size `B` into an ndarray in size `B x D`,
        where `B` is the batch size and `D` is the dimensionality of the encoding.

        :param content: a 1d array of string type in size `B`
        :return: an ndarray in size `B x D` with the embeddings
        """
        import torch

        with torch.no_grad():

            if not self.tokenizer.pad_token:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.model.resize_token_embeddings(len(self.tokenizer.vocab))

            input_tokens = self.tokenizer(
                list(content),
                max_length=self.max_length,
                padding='longest',
                truncation=True,
                return_tensors='pt',
            )
            input_tokens = {k: v.to(self.device) for k, v in input_tokens.items()}

            if self.api_token is not None:
                outputs = self._api_call(list(content))

                hidden_states = torch.tensor([outputs])
            else:
                with self.amp_accelerate():
                    outputs = getattr(self.model, self.embedding_fn_name)(
                        **input_tokens
                    )
                    if isinstance(outputs, torch.Tensor):
                        return outputs.cpu().numpy()
                    hidden_states = outputs.hidden_states

            return self._compute_embedding(hidden_states, input_tokens)
