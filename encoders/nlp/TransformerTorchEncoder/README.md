# TransformerTorchEncoder

TransformerTorchEncoder wraps the pytorch-version of transformers from huggingface, encodes `Document` content from an array of string in size `B` into an ndarray in size `B x D`

## Using the huggingface [API](https://api-inference.huggingface.co/docs/python/html/index.html)

It is also possible to directly call the huggingface API instead of running the model yourself.
The only change needed in the code is adding an `api_token`, e.g.

```
!TransformerTorchEncoder
with:
  pooling_strategy: auto
  pretrained_model_name_or_path: distilbert-base-cased
  max_length: 192
  api_token: api_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```
