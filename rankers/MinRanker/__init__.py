from jina.executors.rankers import Chunk2DocRanker


class MinRanker(Chunk2DocRanker):
    """
    :class:`MinRanker` calculates the score of the matched doc from the matched chunks. For each matched doc, the score
        is `1 / (1 + s)`, where `s` is the minimal score from all the matched chunks belonging to this doc.

    .. warning:: Here we suppose that the smaller chunk score means the more similar.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import warnings
        warnings.warn("MinRanker is deprecated. Please use SimpleAggregateRanker instead", DeprecationWarning,
                      stacklevel=2)

    def _get_score(self, match_idx, query_chunk_meta, match_chunk_meta, *args, **kwargs):
        return self.get_doc_id(match_idx), 1. / (1. + match_idx[self.COL_SCORE].min())
