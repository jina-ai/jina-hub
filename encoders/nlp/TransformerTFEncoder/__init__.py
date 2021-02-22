__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
from typing import Optional

import numpy as np
from jina.executors.decorators import batching, as_ndarray
from jina.executors.devices import TFDevice
from jina.executors.encoders import BaseEncoder


class TransformerTFEncoder(TFDevice, BaseEncoder):
    """
    Internally wraps the tensorflow-version of transformers from huggingface.

    :param pretrained_model_name_or_path: Either:
        - a string, the `model id` of a pretrained model hosted inside a
            model repo on huggingface.co, e.g.: ``bert-base-uncased``.
        - a path to a `directory` containing model weights saved using
            :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.:
            ``./my_model_directory/``.
    :param base_tokenizer_model: The name of the base model to use for
        creating the tokenizer. If None, will be equal to
        `pretrained_model_name_or_path`.
    :param pooling_strategy: the strategy to merge the word embeddings
        into the chunk embedding. Supported strategies include
        'cls', 'mean', 'max', 'min'.
    :param layer_index: index of the transformer layer that is used to
        create encodings. Layer 0 corresponds to the embeddings layer
    :param max_length: the max length to truncate the tokenized sequences to.
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = 'distilbert-base-uncased',
        base_tokenizer_model: Optional[str] = None,
        pooling_strategy: str = 'mean',
        layer_index: int = -1,
        max_length: Optional[int] = None,
        *args,
        **kwargs,
    ):
        """Set Constructor."""
        super().__init__(*args, **kwargs)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.base_tokenizer_model = base_tokenizer_model or pretrained_model_name_or_path
        self.pooling_strategy = pooling_strategy
        self.layer_index = layer_index
        self.max_length = max_length

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

    def post_init(self):
        from transformers import TFAutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_tokenizer_model)
        self.model = TFAutoModel.from_pretrained(
            self.pretrained_model_name_or_path, output_hidden_states=True
        )
        self.to_device()

    @batching
    @as_ndarray
    def encode(self, data: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':
        """
        Encode an array of string in size `B` into an ndarray in size `B x D`

        The ndarray potentially is BatchSize x (Channel x Height x Width)

        :param data: a 1d array of string type in size `B`
        :return: an ndarray in size `B x D`
        """
        import tensorflow as tf

        if not self.tokenizer.pad_token:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer.vocab))

        input_tokens = self.tokenizer(
            list(data),
            max_length=self.max_length,
            padding='longest',
            truncation=True,
            return_tensors='tf',
        )

        outputs = self.model(**input_tokens)

        n_layers = len(outputs.hidden_states)
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
        fill_val = tf.constant(fill_vals[self.pooling_strategy])

        layer = outputs.hidden_states[self.layer_index]
        attn_mask = tf.expand_dims(input_tokens['attention_mask'], -1)
        attn_mask = tf.broadcast_to(attn_mask, layer.shape)
        layer = tf.where(attn_mask == 1, layer, fill_val)

        if self.pooling_strategy == 'cls':
            CLS = self.tokenizer.cls_token_id
            ind = tf.experimental.numpy.nonzero(input_tokens['input_ids'] == CLS)
            output = tf.gather_nd(layer, tf.stack(ind, axis=1))
        elif self.pooling_strategy == 'mean':
            output = tf.reduce_sum(layer, 1) / tf.reduce_sum(tf.cast(attn_mask, tf.float32), 1)
        elif self.pooling_strategy == 'max':
            output = tf.reduce_max(layer, 1)
        elif self.pooling_strategy == 'min':
            output = tf.reduce_min(layer, 1)

        return output.numpy()
