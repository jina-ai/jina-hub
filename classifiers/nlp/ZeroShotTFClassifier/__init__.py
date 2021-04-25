import numpy as np

from typing import List, Optional

from jina.executors.classifiers import BaseClassifier
from jina.executors.decorators import batching, as_ndarray
from jina.executors.devices import TFDevice


class ZeroShotTFClassifier(TFDevice, BaseClassifier):
    """
    :class: 'ZeroShotTFClassifier' wraps tensorflow transformers from
        huggingface, and performs zero shot learning classification
        for nlp data.

    :param labels: the potential labels for the classification
        task.
    :param pretrained_model_name_or_path: Either:
        - a string, the 'model id' of a pretrained model hosted inside a
            model repo on huggingface.co, e.g.: 'bert-base-uncased'.
        - a path to a 'directory' containing model weights saved using
            :func: '~transformers.PreTrainedModel.save_pretrained',
            e.g.: './my_model_directory/'.
    :param base_tokenizer_model: The name of the base model to use for
        creating the tokenizer. If None, will be equal to
        'pretrained_model_name_or_path'.
    :param pooling_strategy: the strategy to merge the word embeddings
        into the chunk embedding. Supported strategies include 'cls',
        'mean', 'max', 'min'.
    :param layer_index: index of the transformer layer that is used
        to create encodings. Layer 0 corresponds to the embeddings layer
    :param max_length: the max length to truncate the tokenized
        sequences to.
    """

    def __init__(
        self,
        labels: List[str],
        pretrained_model_name_or_path: str = 'distilbert-base-uncased',
        base_tokenizer_model: Optional[str] = None,
        pooling_strategy: str = 'mean',
        layer_index: int = -1,
        max_length: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.labels = labels
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.base_tokenizer_model = (
            base_tokenizer_model or pretrained_model_name_or_path
        )
        self.pooling_strategy = pooling_strategy
        self.layer_index = layer_index
        self.max_length = max_length

        if self.pooling_strategy == 'auto':
            self.pooling_strategy = 'cls'
            self.logger.warning(
                '`auto` pooling_strategy is deprecated, Defaulting to '
                ' `cls` to maintain the old default behavior.'
            )

        if self.pooling_strategy not in ['cls', 'mean', 'max', 'min']:
            self.logger.error(
                f'pooling strategy not found: {self.pooling_strategy}.'
                ' The allowed pooling strategies are'
                ' `cls`, `mean`, `max`, `min`.'
            )
            raise NotImplementedError

        if len(self.labels) < 2:
            raise ValueError('The number of target labels must be at least 2.')

        if len(self.labels) != \
                len(set(self.labels)):
            raise ValueError(
                'There are duplicate value in the target_label argument.'
            )

    @as_ndarray
    @batching
    def predict(self,
                content: 'np.ndarray',
                *args,
                **kwargs) -> 'np.ndarray':
        """
         Perform zero shot classification on ``Document`` content, the predicted label
         for each sample in X is returned.

         The output is a zero/one one-hot label for L-class multi-class
         classification of size (B, L) with 'B' being 'content.shape[0]'
         and 'L' being the number of potential classification labels.

         :param content: the input textual data to be classified, a 1 d
            array of string type in size 'B'
         :type content: np.ndarray
         :return: zero/one one-hot predicted label of each sample
            in size '(B, L)'
         :rtype: np.ndarray
        """

        data_encoded = self._encode(content)

        distances = self._evaluate(data_encoded, self.labels_encoded)

        labels_pred = (distances == distances.min(axis=1)[:, None]).astype(int)

        return labels_pred

    def post_init(self):
        from transformers import TFAutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_tokenizer_model
        )
        self.model = TFAutoModel.from_pretrained(
            self.pretrained_model_name_or_path, output_hidden_states=True
        )
        self.to_device()

        self.labels_encoded = self._encode(self.labels)

    def _evaluate(self,
                  actual: 'np.ndarray',
                  desired: 'np.ndarray') -> 'np.ndarray':

        actual = _expand_vector(actual)
        desired = _expand_vector(desired)

        return _cosine(_ext_A(_norm(actual)), _ext_B(_norm(desired)))

    def _encode(self, content: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':

        import tensorflow as tf

        if not self.tokenizer.pad_token:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer.vocab))

        input_tokens = self.tokenizer(
            list(content),
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
                f' valid values are integers from {-n_layers} '
                f' to {n_layers - 1}.'
            )
            raise ValueError

        if self.pooling_strategy == 'cls' and not self.tokenizer.cls_token:
            self.logger.error(
                f'You have set pooling_strategy to `cls`, but the tokenizer'
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
            ind = tf.experimental.numpy.nonzero(
                input_tokens['input_ids'] == CLS
            )
            output = tf.gather_nd(layer, tf.stack(ind, axis=1))
        elif self.pooling_strategy == 'mean':
            output = tf.reduce_sum(layer, 1) / tf.reduce_sum(
                tf.cast(attn_mask, tf.float32), 1
            )
        elif self.pooling_strategy == 'max':
            output = tf.reduce_max(layer, 1)
        elif self.pooling_strategy == 'min':
            output = tf.reduce_min(layer, 1)

        return output.numpy()


def _cosine(A_norm_ext, B_norm_ext):

    return A_norm_ext.dot(B_norm_ext).clip(min=0) / 2


def _ext_A(A):
    nA, dim = A.shape
    A_ext = _get_ones(nA, dim * 3)
    A_ext[:, dim: 2 * dim] = A
    A_ext[:, 2 * dim:] = A ** 2
    return A_ext


def _ext_B(B):
    nB, dim = B.shape
    B_ext = _get_ones(dim * 3, nB)
    B_ext[:dim] = (B ** 2).T
    B_ext[dim: 2 * dim] = -2.0 * B.T
    del B
    return B_ext


def _euclidean(A_ext, B_ext):
    sqdist = A_ext.dot(B_ext).clip(min=0)
    return np.sqrt(sqdist)


def _norm(A):
    return A / np.linalg.norm(A, ord=2, axis=1, keepdims=True)


def _get_ones(x, y):
    return np.ones((x, y))


def _expand_vector(vec):
    if not isinstance(vec, np.ndarray):
        vec = np.array(vec)
    if len(vec.shape) == 1:
        vec = np.expand_dims(vec, 0)
    return vec
