# TFIDFTextEncoder

 **TFIDFTextEncoder** is a class that wraps the text embedding functionality from a **TFIDF** model.

The **TFIDF** model is a classic vector representation for  [information retrieval](https://en.wikipedia.org/wiki/Tf–idf).

`TfidfTextEncoder` encodes data from a `np.ndarray` of strings and returns a `scipy.csr_matrix` of floating point values.

- Input shape: `BatchSize `

- Output shape: `BatchSize x EmbeddingDimension`

    

## Encode with the encoder:

The following example shows how to generate output embeddings given an input `np.ndarray` of strings.

```python
# Input data
text_batch = np.array(['Han likes eating pizza', 'Han likes pizza', 'Jina rocks'])

# Encoder embedding 
encoder = TFIDFTextEncoder()
embeddeding_batch_csr = encoder.encode(text_batch)
```

