# NLTKSentencizer

`NLTKSentencizer` segments text into sentences. NLTK tokenizers are language-specific, which allows them to capture some 
of the language-dependent rules for sentence-level tokenization. NLTKSentencizer uses the same tokenizer used in 
[nltk.tokenize.sent_tokenize](https://www.nltk.org/api/nltk.tokenize.html#nltk.tokenize.sent_tokenize), which is of the class nltk.PunktSentenceTokenizer. 

[nltk.PunktSentenceTokenizer](https://www.nltk.org/api/nltk.tokenize.html#nltk.tokenize.punkt.PunktSentenceTokenizer) uses an unsupervised algorithm to build a model for abbreviation words, collocations, 
and words that start sentences; and then uses that model to find sentence boundaries. This approach has been shown 
to work well for many European languages. Currently, the supported languages are Czech, Danish, Dutch, English, Estonian, Finnish, French, German, Greek, 
Italian, Norwegian, Polish, Portuguese, Russian, Slovene, Spanish, Swedish, Turkish. If no language is specified, 
`NLTKSentencizer` will use English, which may lead to sub-optimal behavior when used with text written in a different 
language. 




## Usage:
The following code snippets show how to use NLTKSentencizer as a segmenter.

- Simple Python usage:

 - ```python
    from jina.hub.segmenters.nlp.NLTKSentencizer import NLTKSentencizer

    sentencizer = NLTKSentencizer()  
    text = "Today is a good day. Can't wait for tomorrow!"
    sentencizer.segment(text)
    # [{'text': 'Today is a good day.', 'offset': 0, 'location': [0, 20]}, 
    # {'text': "Can't wait for tomorrow!", 'offset': 1, 'location': [21, 45]}]
    ```
       

Users can use Pod images in several ways:

- Run with Docker (`docker run`)
  - ```bash
    docker run jinahub/pod.segmenter.nltksentencizer:0.0.1-1.0.1 --port-in 55555 --port-out 55556
    ```
    
- Flow API
  - ```python
    from jina.flow import Flow
      
    def print_chunks(req):
        print("-----------------------")
        for chunk in req.docs[0].chunks:
            print(chunk.text)
        print("-----------------------")
    
    #It may take some time if you don't pull the image, you can set timeout_ready=-1 or pull image locally before.
    f = Flow().add(name='my_segmenter', uses='docker://jinahub/pod.segmenter.nltksentencizer:0.0.1-1.0.1', port_in=55555, port_out=55556, timeout_ready=-1)
    with f:
        f.index_lines(['It is a sunny day!!!! When Andy comes back, we are going to the zoo.'], on_done=print_chunks,  line_format='csv')
    ```
    The `nltksentencizer.yml` can be created with following configurations:
    
    ```yaml
    !NLTKSentencizer
    with:
      language: "english"
    metas:
      py_modules:
        - __init__.py
    ```
- Jina CLI
  - ```bash
    jina pod --uses docker://jinahub/pod.segmenter.nltksentencizer:0.0.1-1.0.1 --port-in 55555 --port-out 55556
    ```
    
- Conventional local usage with `uses` argument, you need to create the YAML file first. You may also want to refer [YAML Syntax](https://docs.jina.ai/chapters/yaml/executor.html).
  - ```bash
    jina pod --uses nltksentencizer.yml --port-in 55555 --port-out 55556
    ```
    
- Docker command

  - Specify the image name along with the version tag. The snippet below uses Jina version `1.0.1`

  - ```bash
    docker pull jinahub/pod.segmenter.nltksentencizer:0.0.1-1.0.1
    ```
   
 Note:
 
 One of the limitations with the Hub Executors currently is the tags - all Executor images should have the versions appended in the name i.e.
 if the version is `0.0.1-1.0.1`, the image name would be `jinahub/pod.segmenter.nltksentencizer:0.0.1-1.0.1`.
   
 