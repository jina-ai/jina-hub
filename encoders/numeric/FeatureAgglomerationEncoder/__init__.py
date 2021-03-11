__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from jina.executors.encoders.numeric import TransformEncoder

class FeatureAgglomerationEncoder(TransformEncoder):
    """
    Encode data agglomerating features.

    Recursively merges features that minimally increases a given linkage distance.
    Similar to AgglomerativeClustering, but recursively merges features instead of samples.

    Encodes data from an ndarray in size `B x T` into an ndarray in size `B x D`
    Where `B` is the batch's size and `T` and `D` are the dimensions pre (`T`)
    and after (`D`) the encoding.

    See more `at <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.FeatureAgglomeration.html>`_
    """

    def post_init(self):
        """Load FeatureAgglomeration model"""
        super().post_init()
        if not self.model:
            from sklearn.cluster import FeatureAgglomeration
            self.model = FeatureAgglomeration(n_clusters=self.output_dim)
