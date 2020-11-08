from jina.executors.rankers import Chunk2DocRanker
import numpy as np

class AggregateRanker(Chunk2DocRanker):
    """
    :class:`AggregateRanker` calculates the score of the matched doc form the matched chunks. For each matched doc, the
    score is aggregated from all the matched chunks belonging to that doc.
    """

    AGGREGATE_FUNCTIONS = ['min', 'max', 'mean', 'median', 'sum', 'prod']

    def __init__(self, aggregate_function: str, is_reversed_score, *args, **kwargs):
        """
        :param: aggregate_function: defines the used aggregate function and can be one of:
        [min, max, mean, median, sum, prod]
        :param: is_reverse_score: True if a large score of the matched chunks mean a lower score of the matched doc,
        othewise: False

        """
        super().__init__(*args, **kwargs)
        self.is_reversed_score = is_reversed_score
        self.aggregate_function = aggregate_function

    def _get_score(self, match_idx, query_chunk_meta, match_chunk_meta, *args, **kwargs):
        scores = match_idx[self.COL_SCORE]
        if self.aggregate_function in self.AGGREGATE_FUNCTIONS:
            aggregated_score = getattr(np, self.aggregate_function)(scores)
        else:
            raise ValueError(f'The aggregate function "{self.aggregate_function}" is not defined.')
        if self.is_reversed_score:
            aggregated_score = 1. / (1. + aggregated_score)
        return self.get_doc_id(match_idx), aggregated_score
