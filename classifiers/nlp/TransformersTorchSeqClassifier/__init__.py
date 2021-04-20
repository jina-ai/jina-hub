import numpy as np
from jina.executors.classifiers import BaseClassifier
from jina.executors.decorators import batching, as_ndarray
from jina.executors.devices import TorchDevice


class TransformersTorchSeqClassifier(TorchDevice, BaseClassifier):
    """
    :class:`TransformersTorchSeqClassifier`
    Wrapper for :class:`transformers.AutoModelForSequenceClassification`. Works with all trained sequence classification models
    on `Huggingface Transformers model hub <https://huggingface.co/models?pipeline_tag=text-classification>`_.

    :param pretrained_model_name_or_path: Default: ``'distilbert-base-uncased-finetuned-sst-2-english'``. Can be either:

        - a string, the `model id` of a pretrained model hosted
            inside a model repo on huggingface.co, e.g.: ``bert-base-uncased``.
        - a path to a `directory` containing model weights saved using
            :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.:
            ``./my_model_directory/``.

    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = 'distilbert-base-uncased-finetuned-sst-2-english',
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        # self.max_length = max_length

    def post_init(self):
        """Load the transformer model and tokenizer"""
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name_or_path
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_model_name_or_path
        )
        self.to_device(self.model)

    @batching
    @as_ndarray
    def predict(self, content: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':
        """
        Performs prediction with sequence classification model
        The size and type of output can be one of the follows (depending on the model used in initializing the class) ``B`` is ``content.shape[0]``:

            - (B,) or (B, 1); float
            - (B, L): soft label for L-class multi-class classification

        :param content: the input textual data to be classified, a 1 d
            array of string type in size `B`
        :type content: np.ndarray
        :param args:  Additional positional arguments
        :param kwargs: Additional keyword arguments
        :return: logits
        :rtype: np.ndarray
        """
        import torch

        inputs = self.tokenizer(
            list(content),
            add_special_tokens=True,
            return_tensors='pt',
            padding='longest',
            truncation=True,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)[0]

        if self.model.config.num_labels == 1:
            scores = 1.0 / (1.0 + np.exp(-outputs))
        else:
            scores = np.exp(outputs) / np.exp(outputs).sum(-1, keepdims=True)

        scores = scores.detach().numpy()
        return scores
