# ImageNormalizer

**ImageNormalizer** is a crafter that normalizes an input image.

The **ImageNormalizer** executor needs the following parameters:

## Snippets:

Initialise ImageNormalizer:

| `param_name`  | `param_remarks` |
| ------------- | ------------- |
| `target_size`  |It's the desired output size.  |
| `img_mean`  |It's the mean of the images in `RGB` channels.  |
| `img_std`  |It's the std of the images in `RGB` channels.  |
| `resize_dim`  |It's the desired size that images will be resized to. They are resized before the cropping |
| `channel_axis`  |It's the axis id of the color channel.  |

**NOTE**: 

- `MODULE_VERSION` is the version of the ImageNormalizer, in semver format. E.g. `0.0.13`.
- `JINA_VERSION` is the version of the Jina core version with which the Docker image was built. E.g. `1.0.1` 

- Flow API

  ```python
    from jina.flow import Flow
    f = (Flow()
        .add(name='my-crafter', uses='docker://jinahub/pod.crafter.imagenormalizer:MODULE_VERSION-JINA_VERSION')
    ```
- Flow YAML file
  This is the only way to provide arguments to its parameters:
  
  ```yaml
  pods:
    - name: ngt
      uses: crafters/image/ImageNormalizer/config.yml
  ```
  
  and then in `imagenormaliser.yml`:
  ```yaml
  !ImageNormalizer
  metas:
    - py_modules:
        - __init__.py
  ```
- Jina CLI
  
  ```bash
  jina pod --uses docker://jinahub/pod.crafter.imagenormalizer:MODULE_VERSION-JINA_VERSION
  ```
- Conventional local usage with `uses` argument
  
  ```bash
  jina pod --uses crafters/image/ImageNormalizer/config.yml --port-in 55555 --port-out 55556
  ```
- Run with Docker (`docker run`)
 
  Specify the image name along with the version tag. The snippet below uses Jina version as `JINA_VERSION`.
  ```bash
    docker run --network host docker://jinahub/pod.crafter.imagenormalizer:MODULE_VERSION-JINA_VERSION --port-in 55555 --port-out 55556
    ```