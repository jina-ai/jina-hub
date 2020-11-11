from jina.executors.rankers import Chunk2DocRanker
import numpy as np

class SimpleAggregateRanker(Chunk2DocRanker):
    """
    :class:`SimpleAggregateRanker` aggregates the score of the matched doc from the matched chunks.
    For each matched doc, the score is aggregated from all the matched chunks belonging to that doc.
    """

    AGGREGATE_FUNCTIONS = ['min', 'max', 'mean', 'median', 'sum', 'prod']

    def __init__(self, aggregate_function: str, is_reversed_score: bool, *args, **kwargs):
        """
        :param: aggregate_function: defines the used aggregate function and can be one of:
        [min, max, mean, median, sum, prod]
        :param: is_reverse_score: True if a large score of the matched chunks mean a lower score of the matched doc,
        othewise: False

        """
        super().__init__(*args, **kwargs)
        self.is_reversed_score = is_reversed_score
        if aggregate_function in self.AGGREGATE_FUNCTIONS:
            self.np_aggregate_function = getattr(np, aggregate_function)
        else:
            raise ValueError(f'The aggregate function "{aggregate_function}" is not in "{self.AGGREGATE_FUNCTIONS}".')

    def _get_score(self, match_idx, query_chunk_meta, match_chunk_meta, *args, **kwargs):
        scores = match_idx[self.COL_SCORE]
        aggregated_score = self.np_aggregate_function(scores)
        if self.is_reversed_score:
            if aggregated_score == -1:
                raise ValueError(f'Setting "is_reversed_score" to True does not allow to have an aggregated document '
                                 f'score of -1.')
            aggregated_score = 1. / (1. + aggregated_score)
        return self.get_doc_id(match_idx), aggregated_score
