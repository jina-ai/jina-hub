# Sentencizer

Sentencizer is one segmenter used in the NLP domain. It splits the text on the doc-level into sentences on the chunk-level with a rule-base strategy.

## Usage:
The Sentencizer is a rule based segmenter which will split text into sentences using '!', '.', '?', etc. as punctuation characters. The following code snippets show how to use it as a segmenter.

Users can use Pod images in several ways:

- Run with Docker (`docker run`)
  - ```bash
    docker run jinahub/pod.segmenter.sentencizer:0.0.9-1.0.1 --port-in 55555 --port-out 55556
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
    f = Flow().add(name='my_segmenter', uses='docker://jinahub/pod.segmenter.sentencizer:0.0.9-1.0.1', port_in=55555, port_out=55556, timeout_ready=-1)
    with f:
        f.index_lines(['  This ,  text is...  . Amazing !!'], on_done=print_chunks,  line_format='csv')
    ```
    
- Jina CLI
  - ```bash
    jina pod --uses docker://jinahub/pod.segmenter.sentencizer:0.0.9-1.0.1 --port-in 55555 --port-out 55556
    ```
    
- Conventional local usage with `uses` argument, you need to create the YAML file first. You may also want to refer [YAML Syntax](https://docs.jina.ai/chapters/yaml/executor.html).
  - ```bash
    jina pod --uses sentencizer.yml --port-in 55555 --port-out 55556
    ```
    
- Docker command

  - Specify the image name along with the version tag. The snippet below uses Jina version `1.0.1`

  - ```bash
    docker pull jinahub/pod.segmenter.sentencizer:0.0.9-1.0.1
    ```
   
 Note:
 
 One of the limitations with the Hub Executors currently is the tags - all Executor images should have the versions appended in the name i.e.
 if the version is `0.0.9-1.0.1`, the image name would be `jinahub/pod.segmenter.sentencizer:0.0.9-1.0.1`.
   
