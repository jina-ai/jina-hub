# TFIDFTextEncoder

 **TFIDFTextEncoder** is a class that wraps the text embedding functionality from a **TFIDF** model.

The **TFIDF** model is a classic vector representation for  [information retrieval](https://en.wikipedia.org/wiki/Tf–idf).

`TfidfTextEncoder` encodes data from a `np.ndarray` of strings and returns a `scipy.csr_matrix` of floating point values.

- Input shape: `BatchSize `

- Output shape: `BatchSize x EmbeddingDimension`




## How to 

### Dependencies

`TFIDFTextEncoder` is dependend on `sckikit-learn`:

```
pip install scikit-learn
```



### Use as Python Class

Before the `TFIDFTextEncoder` is used a `sklearn.feature_extraction.text.TfidfVectorizer` object needs to be fitted and stored as a pickle object which the `TFIDFTextEncoder` will load. 

The following snipped can be used to fit a `TfidfVectorizer` with a toy corpus from sklearn. To achieve better performance or adapt the encoder to other languages users can change the `load_data` function from below to load any other user specific dataset.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def load_data():
    from sklearn.datasets import fetch_20newsgroups
    newsgroups_train = fetch_20newsgroups(subset='train')
    return newsgroups_train.data

if __name__ == '__main__':
    X = load_data()    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(X)
    pickle.dump(tfidf_vectorizer, open('tfidf_vectorizer.pickle', 'wb'))
```



The following example shows how to generate output embeddings given an input `np.ndarray` of strings.

```python
# Input data
text_batch = np.array(['Han likes eating pizza', 'Han likes pizza', 'Jina rocks'])

# Encoder embedding 
encoder = TFIDFTextEncoder(path_vectorizer= './model/tfidf_vectorizer.pickle')
embeddeding_batch_csr = encoder.encode(text_batch)
```

Here `embeddeding_batch_csr` will be a `scipy.sparse.csr_matrix` object containing in each row the embedded TF-IDF vectors.



### Use in Flow API

- `MODULE_VERSION` is the version of the `TFIDFTextEncoder`, in semver format. E.g. `0.0.16`.
- `JINA_VERSION` is the version of the Jina core version with which the Docker image was built. E.g. `1.0.7` 

- Flow API

```python
from jina.flow import Flow
f = (Flow().add(name='my-encoder', uses='docker://jinahub/pod.encoder.tfidftextencoder:MODULE_VERSION-JINA_VERSION')
```

- Flow YAML file
    This is the only way to provide arguments to its parameters:

    ```yaml
    pods:
      - name: tfidf
        uses: encoders/nlp/TFIDFTextEncoder/config.yml
    ```

    and then in `tfidftextencoder.yml`:

    ```yaml
    !TFIDFTextEncoder
    ```

- Jina CLI

    ```bash
    jina pod --uses docker://jinahub/pod.encoders.nlp.tfidftextencoder:MODULE_VERSION-JINA_VERSION
    ```

- Conventional local usage with `uses` argument

    ```bash
    jina pod --uses encoders/nlp/TFIDFTextEncoder/config.yml --port-in 55555 --port-out 55556
    ```

- Run with Docker (`docker run`)

    Specify the image name along with the version tag. The snippet below uses Jina version as `JINA_VERSION`.

    ```bash
      docker run --network host docker://jinahub/pod.encoders.tfidftextencoder:MODULE_VERSION-JINA_VERSION --port-in 55555 --port-out 55556
    ```

