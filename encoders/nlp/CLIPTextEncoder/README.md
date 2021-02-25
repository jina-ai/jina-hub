# ClipTextEncoder

 **CliipTextEncoder** is a class that wraps the text embedding functionality from the **CLIP** model.

The **CLIP** model was originally proposed in  [Learning Transferable Visual Models From Natural Language Supervision](https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf).

`ClipTextEncoder` encodes data from a `np.ndarray` of strings and returns a `np.ndarray` of floating point values.

- Input shape: `BatchSize `

- Output shape: `BatchSize x EmbeddingDimension`

    

## Encode with the encoder:

The following example shows how to generate output embeddings given an input `np.ndarray` of strings.

```python
# Input data
text_batch = np.array(['Han likes eating pizza', 'Han likes pizza', 'Jina rocks'])

# Encoder embedding 
encoder = CLIPTextEncoder()
embeddeding_batch_np = encoder.encode(text_batch)
```

