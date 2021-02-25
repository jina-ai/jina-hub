# ClipImageEncoder

 **CliipImageEncoder** is a class that wraps the image embedding functionality from the **CLIP** model.

The **CLIP** model was originally proposed in  [Learning Transferable Visual Models From Natural Language Supervision](https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf).

`ClipImageEncoder` encodes data from a `np.ndarray` of floats and returns a `np.ndarray` of floats.

- Input shape: `BatchSize x (Channel x Height x Width)`

- Output shape: `BatchSize x EmbeddingDimension`

      

## Encode with the encoder:

The following example shows how to generate output embeddings given an input `np.ndarray` containing images.

```python
# Encoder embedding 
encoder = CLIPImageEncoder()
embeddeding_batch_np = encoder.encode(batch_of_images)    
```

