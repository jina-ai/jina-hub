# SpacySentencizer

SpacySentencizer is one segmenter used in the NLP domain. It splits the text on the doc-level into sentences on the chunk-level with spaCy as the backend.

## Usage:
**Users are expected to download spaCy model before using this segmenter. To download the model, please refer to this page https://spacy.io/usage/models**
The splitting technique is divided into two different approaches:
1) Default Segmenter: Utilizes dependency parsing to determine the rule for splitting the text. For more info please refer to https://spacy.io/api/sentencizer.
2) Machine Learning-based Segmenter: Utilizes SentenceRecognizer model trained on a dedicated data created for sentence segmentation. For more info please refer to https://spacy.io/api/sentencerecognizer.

The following code snippets show how to use it as a segmenter.

Users can use Pod images in several ways:

- Run with Docker (`docker run`)
  - ```bash
    docker run jinahub/pod.segmenter.spacysentencizer:0.0.1-1.0.1 --port-in 55555 --port-out 55556
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
    f = Flow().add(name='my_segmenter', uses='docker://jinahub/pod.segmenter.spacysentencizer:0.0.1-1.0.1', port_in=55555, port_out=55556, timeout_ready=-1)
    with f:
        f.index_lines(['It is a sunny day!!!! When Andy comes back, we are going to the zoo.'], on_done=print_chunks,  line_format='csv')
    ```
    The `spacysentencizer.yml` can be created with following configurations:
    
    ```yaml
    !SpacySentencizer
    with:
      lang: "xx_sent_ud_sm"
      use_default_segmenter: false
    metas:
      py_modules:
        - __init__.py
    ```
- Jina CLI
  - ```bash
    jina pod --uses docker://jinahub/pod.segmenter.spacysentencizer:0.0.1-1.0.1 --port-in 55555 --port-out 55556
    ```
    
- Conventional local usage with `uses` argument, you need to create the YAML file first. You may also want to refer [YAML Syntax](https://docs.jina.ai/chapters/yaml/executor.html).
  - ```bash
    jina pod --uses sentencizer.yml --port-in 55555 --port-out 55556
    ```
    
- Docker command

  - Specify the image name along with the version tag. The snippet below uses Jina version `1.0.1`

  - ```bash
    docker pull jinahub/pod.segmenter.spacysentencizer:0.0.1-1.0.1
    ```
   
 Note:
 
 One of the limitations with the Hub Executors currently is the tags - all Executor images should have the versions appended in the name i.e.
 if the version is `0.0.1-1.0.1`, the image name would be `jinahub/pod.segmenter.spacysentencizer:0.0.1-1.0.1`.
   
