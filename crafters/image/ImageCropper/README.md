# ImageCropper

ImageCropper crops the image with the specific crop box. The coordinate is the same coordinate-system.

## Snippets:

Initialise ImageCropper:

`ImageCropper(top, left, height, width, channel_axis)`
| File              | Descriptions                                                 |
|-------------------|--------------------------------------------------------------|
| `top`             | Vertical coordinate of the top left corner of the crop box   |
| `left`            | Horizontal coordinate of the top left corner of the crop box |
| `height`          | Height of the crop box                                       |
| `width`           | Width of the crop box                                        |
| `channel_axis`    | Axis referring to the channels                               |


**NOTE**: 

- `MODULE_VERSION` is the version of the ImageCropper, in semver format. E.g. `0.0.13`.
- `JINA_VERSION` is the version of the Jina core version with which the Docker image was built. E.g. `1.0.4` 

- Flow API

  ```python
    from jina.flow import Flow
    f = (Flow()
        .add(name='my-crafter', uses='docker://jinahub/pod.crafter.imagecropper:MODULE_VERSION-JINA_VERSION')
    ```
- Flow YAML file
  This is the only way to provide arguments to its parameters:
  
  ```yaml
  pods:
    - name: ngt
      uses: crafters/image/ImageCropper/config.yml
  ```
  
  and then in `imagecropper.yml`:
  ```yaml
  !ImageCropper
  metas:
    - py_modules:
        - __init__.py
  ```
- Jina CLI
  
  ```bash
  jina pod --uses docker://jinahub/pod.crafter.imagecropper:MODULE_VERSION-JINA_VERSION
  ```
- Conventional local usage with `uses` argument
  
  ```bash
  jina pod --uses crafters/image/ImageCropper/config.yml --port-in 55555 --port-out 55556
  ```
- Run with Docker (`docker run`)
 
  Specify the image name along with the version tag. The snippet below uses Jina version as `JINA_VERSION`.
  ```bash
    docker run --network host docker://jinahub/pod.crafter.imagecropper:MODULE_VERSION-JINA_VERSION --port-in 55555 --port-out 55556
    ```