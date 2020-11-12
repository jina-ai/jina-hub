# VSEImageEncoder

`VSEImageEncoder` is the image Encoder used to extract Visual Semantic Embeddings. Taken from the results of
VSE++: Improving Visual-Semantic Embeddings with Hard Negatives (https://arxiv.org/abs/1707.05612). This model
extracts image feature embeddings that can be used in combination with a VSETextEncoder which in combination will
put images and its captions in nearby locations in the embedding space

@article{faghri2018vse++,
  title={VSE++: Improving Visual-Semantic Embeddings with Hard Negatives},
  author={Faghri, Fartash and Fleet, David J and Kiros, Jamie Ryan and Fidler, Sanja},
  booktitle = {Proceedings of the British Machine Vision Conference ({BMVC})},
  url = {https://github.com/fartashf/vsepp},
  year={2018}
}