# TransformersTorchSeqClassifier

Sequence (text) classification wrapper for [HuggingFace sequence classification models](https://huggingface.co/transformers/usage.html#sequence-classification). 
Works with all trained sequence classification models, including the ones hosted on 
[HuggingFace Transformers model hub](https://huggingface.co/models?pipeline_tag=text-classification).


## Usage:
The following code snippets show how to use TransformersTorchSeqClassifier.

- Simple Python usage:

 - ```python
   from jina.hub.classifiers.nlp.TransformersTorchSeqClassifier import TransformersTorchSeqClassifier
   import numpy as np
   
   model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
   classifier = TransformersTorchSeqClassifier(model_name)
   content =np.stack(['Today is a good day.',
                   "Can't wait for tomorrow!",
                   "Today is a good day. Can't wait for tomorrow!"])
   output = classifier.predict(content)
   print(output.shape)  # (3,2)
   print(output.argmax(axis=1)) # [1,1,1]
   print(classifier.model.config.id2label[output[2,:].argmax()]) # POSITIVE
    ```
       

Users can use Pod images in several ways:

- Run with Docker (`docker run`)
  ```bash
    docker run --network host jinahub/pod.classifier.transformerstorchseqclassifier:0.0.1-1.0.1 --port-in 55555 --port-out 55556
    ```
    
- The `TransformersTorchSeqClassifier.yml` can be created with following configurations:
    ```yaml
      !TransformersTorchSeqClassifier
      with:
        pretrained_model_name_or_path: "distilbert-base-uncased-finetuned-sst-2-english"
      metas:
        py_modules: 
          - __init__.py
        ```
- Jina CLI
  ```bash
    jina pod --uses docker://jinahub/pod.classifier.transformerstorchseqclassifier:0.0.1-1.0.1 --port-in 55555 --port-out 55556
    ```
    
- Conventional local usage with `uses` argument, you need to create the YAML file first. You may also want to refer [YAML Syntax](https://docs.jina.ai/chapters/yaml/executor.html).
  - ```bash
    jina pod --uses transformerstorchseqclassifier.yml --port-in 55555 --port-out 55556
    ```
    
- Docker command

  - Specify the image name along with the version tag. The snippet below uses Jina version `1.0.1`

  - ```bash
    docker pull jinahub/pod.classifier.transformerstorchseqclassifier:0.0.1-1.0.1
    ```
   
 Note:
 
 One of the limitations with the Hub Executors currently is the tags - all Executor images should have the versions appended in the name i.e.
 if the version is `0.0.1-1.0.1`, the image name would be `jinahub/pod.classifier.transformerstorchseqclassifier:0.0.1-1.0.1`.
   
 
