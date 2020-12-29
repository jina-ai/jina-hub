__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
from typing import Optional

import numpy as np
from jina.executors.decorators import batching, as_ndarray
from jina.executors.devices import TFDevice
from jina.executors.encoders import BaseEncoder
from jina.executors.encoders.helper import reduce_mean, reduce_max, reduce_min, reduce_cls
from jina.logging import default_logger


def auto_reduce(model_outputs: 'np.ndarray', mask_2d: 'np.ndarray', model_name: str) -> 'np.ndarray':
    """
    Automatically creates a sentence embedding from its token embeddings.
        * For BERT-like models (BERT, RoBERTa, DistillBERT, Electra ...) uses embedding of first token
        * For XLM and XLNet models uses embedding of last token
        * Assumes that other models are language-model like and uses embedding of last token
    """
    if 'bert' in model_name or 'electra' in model_name:
        return reduce_cls(model_outputs, mask_2d)
    if 'xlnet' in model_name:
        return reduce_cls(model_outputs, mask_2d, cls_pos='tail')
    default_logger.warning('Using embedding of a last token as a sequence embedding. '
                           'If that is not desirable, change `pooling_strategy`')
    return reduce_cls(model_outputs, mask_2d, cls_pos='tail')


class TransformerTFEncoder(TFDevice, BaseEncoder):
    """
    Internally, TransformerTFEncoder wraps the tensorflow-version of transformers from huggingface.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = 'sentence-transformers/distilbert-base-nli-stsb-mean-tokens',
        base_tokenizer_model: Optional[str] = None,
        pooling_strategy: str = 'mean',
        layer_index: int = -1,
        max_length: Optional[int] = None,
        model_save_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        :param pretrained_model_name_or_path: Either:
            - a string, the `model id` of a pretrained model hosted inside a model repo on huggingface.co, e.g.: ``bert-base-uncased``.
            - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
        :param base_tokenizer_model: The name of the base model to use for creating the tokenizer. If None, will be equal to `pretrained_model_name_or_path`.
        :param pooling_strategy: the strategy to merge the word embeddings into the chunk embedding. Supported
            strategies include 'cls', 'mean', 'max', 'min'.
        :param layer_index: index of the transformer layer that is used to create encodings. Layer 0 corresponds to the embeddings layer
        :param max_length: the max length to truncate the tokenized sequences to.
        :param model_save_path: the path of the encoder model. If a valid path is given, the encoder will be saved to the given path

        ..warning::
            `model_save_path` should be relative to executor's workspace
        """
        super().__init__(*args, **kwargs)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.base_tokenizer_model = base_tokenizer_model or pretrained_model_name_or_path
        self.pooling_strategy = pooling_strategy
        self.layer_index = layer_index
        self.max_length = max_length
        self.model_save_path = model_save_path

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

    def __getstate__(self):
        if self.model_save_path:
            if not os.path.exists(self.model_abspath):
                self.logger.info(f'create folder for saving transformer models: {self.model_abspath}')
                os.mkdir(self.model_abspath)
            self.model.save_pretrained(self.model_abspath)
            self.tokenizer.save_pretrained(self.model_abspath)
        return super().__getstate__()

    @property
    def model_abspath(self) -> str:
        """Get the file path of the encoder model storage
        """
        return self.get_file_from_workspace(self.model_save_path)

    @batching
    @as_ndarray
    def encode(self, data: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':
        """
        :param data: a 1d array of string type in size `B`
        :return: an ndarray in size `B x D`
        """
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

        output_embeddings = hidden_states[self.layer_index]
        _mask_ids_batch = input_tokens['attention_mask'].numpy()
        _seq_output = output_embeddings.numpy()

        if self.pooling_strategy == 'cls':
            output = auto_reduce(_seq_output, _mask_ids_batch, self.model.base_model_prefix)
        elif self.pooling_strategy == 'mean':
            output = reduce_mean(_seq_output, _mask_ids_batch)
        elif self.pooling_strategy == 'max':
            output = reduce_max(_seq_output, _mask_ids_batch)
        elif self.pooling_strategy == 'min':
            output = reduce_min(_seq_output, _mask_ids_batch)

        return output
