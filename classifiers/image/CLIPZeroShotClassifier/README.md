# CLIPZeroShotClassifier

 **CLIPZeroShotClassifier** is a class that wraps the embedding functionality from the **CLIP** model and preforms zero shot classification

The **CLIP** model was originally proposed in  [Learning Transferable Visual Models From Natural Language Supervision](https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf).

`CLIPZeroShotClassifier` classifies ``Document`` content from a `np.ndarray` of floats and returns a `np.ndarray`.

- Input shape: `BatchSize x (Channel x Height x Width)`

- Output shape: `BatchSize x Number of Labels`

> Note that CLIPZeroShotClassifier does not use `embedding` from `Document` but depends on the `content` of the `Document`. Make sure you have set `fields` = `content` in the driver as shown below. 

## Classify with the classifier:

The following example shows how to generate output given an input `np.ndarray` containing images.

```python
classifier = CLIPZeroShotClassifier(labels = ["dog","cat","human"])
classifier = classifier.predict(batch_of_images)    
```

## Usage

1. Flow YAML file
  
  ```yaml
  pods:
    - name: clipzeroshotclassifier
      uses: CLIPZeroShotClassifier/config.yml
  ```
 and then in `config.yml`:

  ```yaml
  !CLIPZeroShotClassifier
  with:
    labels: ["dog","cat","human"]
  requests:
    on:
      IndexRequest:
      - !OneHotPredictDriver
        with:
          fields: "content"
          labels: ["dog","cat","human"]
  ```