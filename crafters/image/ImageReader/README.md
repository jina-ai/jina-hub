# ImageReader

**ImageReader** is a crafter that loads an image, either from a given file path or directly from bytes and save the `ndarray` of the image in the Document. It reads the image specified in `buffer` and save the `ndarray` of the image in the `blob` of the document.

The **ImageReader** executor needs only one parameter:

## Snippets:

| `param_name`  | `param_remarks` |
| ------------- | ------------- |
| `channel_axis`  |The axis id of the color channel. The **-1** is the color channel info at the last axis.|

**NOTE**: 

- `MODULE_VERSION` is the version of the ImageReader, in semver format. E.g. `0.0.13`.
- `JINA_VERSION` is the version of the Jina core version with which the Docker image was built. E.g. `1.0.1` 

- Flow API

  ```python
    from jina.flow import Flow
    f = (Flow()
        .add(name='my-crafter', uses='docker://jinahub/pod.crafter.imagereader:MODULE_VERSION-JINA_VERSION')
    ```
- Flow YAML file
  This is the only way to provide arguments to its parameters:
  
  ```yaml
  pods:
    - name: imgreader
      uses: crafters/image/ImageReader/config.yml
  ```
  
  and then in `imagereader.yml`:
  ```yaml
  !ImageReader
  metas:
    - py_modules:
        - __init__.py
  ```
- Jina CLI
  
  ```bash
  jina pod --uses docker://jinahub/pod.crafter.imagereader:MODULE_VERSION-JINA_VERSION
  ```
- Conventional local usage with `uses` argument
  
  ```bash
  jina pod --uses crafters/image/ImageReader/config.yml --port-in 55555 --port-out 55556
  ```
- Run with Docker (`docker run`)
 
  Specify the image name along with the version tag. The snippet below uses Jina version as `JINA_VERSION`.
  ```bash
    docker run --network host docker://jinahub/pod.crafter.imagereader:MODULE_VERSION-JINA_VERSION --port-in 55555 --port-out 55556
    ```