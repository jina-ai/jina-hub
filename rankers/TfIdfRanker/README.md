# TfIdfRanker

`TfIdfRanker` calculates the weighted score from the matched chunks. The weights of each chunk is based on
the tf-idf algorithm. Each query chunk is considered as a ``term``, and the frequency of the query chunk in a
specific matched document is considered as the naive ``term-frequency``. All the matched results as a whole is
considered as the corpus, and therefore the frequency of the query chunk in all the matched docs is considered
as the naive ``document-frequency``. Please refer to the functions for the details of calculating ``tf`` and
``idf``.