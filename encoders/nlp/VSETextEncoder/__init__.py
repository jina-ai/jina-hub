__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import pickle

from jina.executors.encoders.frameworks import BaseTorchEncoder
from jina.executors.decorators import as_ndarray, batching

from .vocab import Vocabulary


class CustomUnpickler(pickle.Unpickler):
    """Custom Unpickler class"""

    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)


class VSETextEncoder(BaseTorchEncoder):
    """
    Encode an 1d array of string in size `B` into an ndarray in size `B x D`

    The ndarray potentially is BatchSize x (Channel x Height x Width)

    Using VSE text_emb branch

    :path : path where to find the model.pth file
    :vocab_path : path where to find the vocab.pkl file
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    """

    def __init__(self, path: str = 'runs/f30k_vse++_vggfull/model_best.pth.tar',
                 vocab_path: str = 'vocab/f30k_vocab.pkl', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = path
        self.vocab_path = vocab_path

    def post_init(self):
        """Load model"""
        import nltk
        import torch
        from .model import VSE

        nltk.download('punkt')
        checkpoint = torch.load(self.path,
                                map_location=torch.device('cpu' if not self.on_gpu else 'cuda'))
        opt = checkpoint['opt']
        with open(self.vocab_path, 'rb') as f:
            self.vocab = CustomUnpickler(f).load()

        opt.vocab_size = len(self.vocab)
        model = VSE(opt)
        model.load_state_dict(checkpoint['model'])
        model.txt_enc.eval()
        self.model = model.txt_enc
        del model.img_enc

    @batching
    def encode(self, content):
        """
        Encodes ``Document`` content from a ndarray of texts into a ndarray of `B x D`.

        :param content: ndarray with texts to be encoded
        :type content: ndarray
        :return: Embedded sentences in ndarray of `B x D`.
        :rtype: ndarray
        """
        import torch
        import nltk
        from torch.autograd import Variable

        captions = []
        for sentence in content:
            tokens = nltk.tokenize.word_tokenize(str(sentence).lower())
            caption = []
            caption.append(self.vocab('<start>'))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab('<end>'))
            target = torch.Tensor(caption)
            captions.append(target)
        lengths = [len(x) for x in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]
        captions_tensor = Variable(targets, requires_grad=False)
        if torch.cuda.is_available():
            captions_tensor = captions_tensor.cuda()
        text_emb = self.model(captions_tensor, lengths=lengths)
        return text_emb.detach().numpy()
