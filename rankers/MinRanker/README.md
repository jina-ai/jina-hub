# MinRanker

`MinRanker` calculates the score of the matched doc form the matched chunks. For each matched doc, the score
        is `1 / (1 + s)`, where `s` is the minimal score from all the matched chunks belonging to this doc.