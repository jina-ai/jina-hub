# ImageResizer

**ImageResizer** is a crafter that resizes the image to the given size.
The **ImageResizer** executor needs below parameters:

## Snippets:

| `param_name`  | `param_remarks` |
| ------------- | ------------- |
| `target_size`  |Desired output size|
| `how`  |The interpolation method i.e. `NEAREST`, `BILINEAR`, `BICUBIC`, and `LANCZOS`|
| `channel_axis`  |The axis id of the color channel. The **-1** is the color channel info at the last axis.|

**NOTE**: 

- `MODULE_VERSION` is the version of the ImageResizer, in semver format. E.g. `0.0.13`.
- `JINA_VERSION` is the version of the Jina core version with which the Docker image was built. E.g. `1.0.1` 


## Usage

Users can use Pod images in several ways:

- Flow API

  ```python
    from jina.flow import Flow
    f = (Flow()
        .add(name='my-crafter', uses='docker://jinahub/pod.crafter.imageresizer:MODULE_VERSION-JINA_VERSION')
    ```
- Flow YAML file
  This is the only way to provide arguments to its parameters:
  
  ```yaml
  pods:
    - name: imgresizer
      uses: crafters/image/ImageResizer/config.yml
  ```
  
  and then in `imageresizer.yml`:
  ```yaml
  !ImageResizer
  metas:
    - py_modules:
        - __init__.py
  ```
- Jina CLI
  
  ```bash
  jina pod --uses docker://jinahub/pod.crafter.imageresizer:MODULE_VERSION-JINA_VERSION
  ```
- Conventional local usage with `uses` argument
  
  ```bash
  jina pod --uses crafters/image/ImageResizer/config.yml --port-in 55555 --port-out 55556
  ```
- Run with Docker (`docker run`)
 
  Specify the image name along with the version tag. The snippet below uses Jina version as `JINA_VERSION`.
  ```bash
    docker run --network host docker://jinahub/pod.crafter.imageresizer:MODULE_VERSION-JINA_VERSION --port-in 55555 --port-out 55556
    ```
