import numpy as np
from jina.executors.rankers import Chunk2DocRanker


class SimpleAggregateRanker(Chunk2DocRanker):
    """
    :class:`SimpleAggregateRanker` aggregates the score
    of the matched doc from the matched chunks.
    For each matched doc, the score is aggregated
    from all the matched chunks belonging to that doc.

    :param: aggregate_function: defines the used aggregate function
        and can be one of: [min, max, mean, median, sum, prod]
    :param: inverse_score: plus-one inverse by 1/(1+score)
    :raises:
        ValueError: If `aggregate_function` is not any of the expected types
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    """

    AGGREGATE_FUNCTIONS = ['min', 'max', 'mean', 'median', 'sum', 'prod']

    def __init__(self, aggregate_function: str, inverse_score: bool = False, *args, **kwargs):
        """Set constructor"""
        super().__init__(*args, **kwargs)
        self.inverse_score = inverse_score
        if aggregate_function in self.AGGREGATE_FUNCTIONS:
            self.np_aggregate_function = getattr(np, aggregate_function)
        else:
            raise ValueError(f'The aggregate function "{aggregate_function}" is not in "{self.AGGREGATE_FUNCTIONS}".')

    def _get_score(self, match_idx, query_chunk_meta, match_chunk_meta, *args, **kwargs):
        scores = match_idx[self.COL_SCORE]
        aggregated_score = self.np_aggregate_function(scores)
        if self.inverse_score:
            if aggregated_score < 0:
                raise ValueError(f'Setting "is_reversed_score" to True does not allow to have an aggregated document '
                                 f'negative score')
            aggregated_score = 1. / (1. + aggregated_score)
        return self.get_doc_id(match_idx), aggregated_score
