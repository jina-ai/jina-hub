from jina.executors.encoders.numeric import TransformEncoder


class UMAPEncoder(TransformEncoder):
    """
    :class:`UMAPEncoder` encodes data using Uniform Manifold Approximation and Projection Embedding.

    Encodes data from an ndarray of size `B x T` into an ndarray of size `B x D`
    Where `B` is the batch's size and `T` and `D` are the dimensions pre (`T`)
    and after (`D`) the encoding.

    Full code and documentation can be found
    `here <https://github.com/lmcinnes/umap>`_.
    """
    def post_init(self):
        """Load UMAP model"""
        super().post_init()
        if not self.model:
            from umap import UMAP
            self.model = UMAP(n_components=self.output_dim, random_state=self.random_state)