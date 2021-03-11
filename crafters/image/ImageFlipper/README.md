# ImageFlipper

Flip the image horizontally or vertically. Flip image in the left/right or up/down direction respectively.

## Snippets:

Initialise ImageFlipper:

`ImageFlipper(vertical, channel_axis)`
| File                            | Descriptions                                                                      |
|---------------------------------|-----------------------------------------------------------------------------------|
| `vertical`                      | Desired rotation type, vertical if true                                           |
| `channel_axis`                  | Axis for color channel, ``-1``  indicates the color channel info at the last axis |


**NOTE**: 

- `MODULE_VERSION` is the version of the ImageFlipper, in semver format. E.g. `0.0.12`.
- `JINA_VERSION` is the version of the Jina core version with which the Docker image was built. E.g. `1.0.4` 

- Flow API

  ```python
    from jina.flow import Flow
    f = (Flow()
        .add(name='my-crafter', uses='docker://jinahub/pod.crafter.imageflipper:MODULE_VERSION-JINA_VERSION')
    ```
- Flow YAML file
  This is the only way to provide arguments to its parameters:
  
  ```yaml
  pods:
    - name: ngt
      uses: crafters/image/ImageFlipper/config.yml
  ```
  
  and then in `imageflipper.yml`:
  ```yaml
  !ImageFlipper
  metas:
    - py_modules:
        - __init__.py
  ```
- Jina CLI
  
  ```bash
  jina pod --uses docker://jinahub/pod.crafter.imageflipper:MODULE_VERSION-JINA_VERSION
  ```
- Conventional local usage with `uses` argument
  
  ```bash
  jina pod --uses crafters/image/ImageFlipper/config.yml --port-in 55555 --port-out 55556
  ```
- Run with Docker (`docker run`)
 
  Specify the image name along with the version tag. The snippet below uses Jina version as `JINA_VERSION`.
  ```bash
    docker run --network host docker://jinahub/pod.crafter.imageflipper:MODULE_VERSION-JINA_VERSION --port-in 55555 --port-out 55556
    ```