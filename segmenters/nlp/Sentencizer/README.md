# Sentencizer

##What is Sentencizer
Sentencizer is one segmenter used in nlp domain. It splits the text on the doc-level into sentences on the chunk-level with a rule-base strategy.

##How to use it in Flow:
The Sentencizer is a rule based segmenter which will split text into sentences using ['!', '.', '?'] as punctuation characters. The following code snippet shows how to use it as a segmenter.
```python
from jina.flow import Flow


def print_chunks(req):
    print("-----------------------")
    for chunk in req.docs[0].chunks:
        print(chunk.text)
    print("-----------------------")


f = Flow().add(uses='!Sentencizer')
with f:
    f.index_lines(['  This ,  text is...  . Amazing !!'], on_done=print_chunks)
```