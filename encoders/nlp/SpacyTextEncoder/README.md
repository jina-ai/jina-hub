# SpacyTextEncoder

**SpacyTextEncoder** is a class that wraps the text embedding functionality from the **Spacy tok2vec** model.
`SpacyTextEncoder` encodes `Document` content from a `np.ndarray` of strings and returns a `np.ndarray` of floating point values.
spaCy is a library for advanced Natural Language Processing in Python and Cython. It's built on the very latest research, and was designed from day one to be used in real products.

- Input shape: `BatchSize `

- Output shape: `BatchSize x EmbeddingDimension`


SpacyTextEncoder is one encoder used in the NLP domain. It encodes the text following `tok2vec` with spaCy as the backend.
`tok2vec` applies a “token-to-vector” model and set its outputs in the `Doc.tensor` attribute.

## Usage:
**Users are expected to download spaCy model before using this encoder. To download the model, please refer to this page https://spacy.io/usage/models**

The following code snippets show how to use it as a encoder.

Users can use Pod images in several ways:

- Run with Docker (`docker run`)
  - ```bash
    docker run jinahub/pod.encoder.spacyencoder:0.0.1-1.0.16 --port-in 55555 --port-out 55556
    ```
    
- Flow API
  - ```python
    from jina.flow import Flow

    f = Flow().add(name='my_encoder', uses='docker://jinahub/pod.encoder.spacyencoder:0.0.1-1.0.16', port_in=55555, port_out=55556, timeout_ready=-1)
    with f:
        f.index_lines(['It is a sunny day!!!! When Andy comes back, we are going to the zoo.'], on_done=print_chunks,  line_format='csv')
    ```
    The `spacyencoder.yml` can be created with following configurations:
    
    ```yaml
    !spacyencoder
    with:
      lang: "en_core_web_sm"
      use_default_encoder: false
    metas:
      py_modules:
        - __init__.py
    ```
    

## Encode with the encoder:

The following example shows how to generate output embeddings given an input `np.ndarray` of strings.

```python
# Input data
text_batch = np.array(['Han likes eating pizza', 'Han likes pizza', 'Jina rocks'])

# Encoder embedding 
encoder = SpacyTextEncoder()
embeddeding_batch_np = encoder.encode(text_batch)
```

- Jina CLI
  - ```bash
    jina pod --uses docker://jinahub/pod.encoder.spacyencoder:0.0.1-1.0.16 --port-in 55555 --port-out 55556
    ```
    
- Conventional local usage with `uses` argument, you need to create the YAML file first. You may also want to refer [YAML Syntax](https://docs.jina.ai/chapters/yaml/executor.html).
  - ```bash
    jina pod --uses sentencizer.yml --port-in 55555 --port-out 55556
    ```
    
- Docker command

  - Specify the image name along with the version tag. The snippet below uses Jina version `1.0.16`

  - ```bash
    docker pull jinahub/pod.encoder.spacyencoder:0.0.1-1.0.16
    ```
