# ZeroShotTFClassifier

**`ZeroShotTFClassifier`** wraps the TensorFlow version of transformers from 
huggingface, and performs zero-shot learning classification for NLP data. 

It takes an ndarray of sequences and return a ndarray of labels.

* Input length: `BatchSize`
* Output shape: `BatchSize x Number of Labels`

## Usage

Initialize this Executor specifying parameters as below:

| `param_name`  | `param_remarks` |
| ------------- | ------------- |
| `labels`  | potentials labels for classification task |
| `pretrained_model_name_or_path`  | either model id on huggingface or path to model weights |
| `base_tokenizer_model`  | name of the base model to use for creating the tokenizer |
| `pooling_strategy` | strategy of pooling operation |
| `layer_index` | index of the transformer layer that is used to create encodings |
| `max_length` | max length to truncate the tokenized sequences to |

The default pretrained model is distilbert-base-uncased` from huggingface.`

### Snippets:

Initialize ZeroShotTFClassifier:

```python
ZeroShotTFClassifier(labels=['label1', 'label2'])
```

Users can use Pod images in several ways:

#### 1. Run with docker

```
docker run jinahub/pod.classifier.zeroshottfclassifier:0.0.8-1.0.4
```

#### 2. Run the Flow API:

```python
from jina.flow import Flow

f = (Flow()
    .add(name='my-classifier', 
         uses='docker://jinahub/pod.classifier.zeroshottfclassifier:0.0.8-1.0.4', 
         port_in=55555, 
         port_out=55556))
```

#### 3. Run with Jina CLI:

```
jina pod --uses docker://jinahub/pod.classifier.zeroshottfclassifier:0.0.8-1.0.4 --port-in 55555 --port-out 55556
```

#### 4. Conventional local usage with ``uses`` argument:

```
jina pod --uses custom_folder/zeroshottfclassifier.yml --port-in 55555 --port-out 55556
```

#### 5. Docker command with specified image name and version tag:

```
docker pull jinahub/pod.classifier.zeroshottfclassifier:0.0.8-1.0.4
```
