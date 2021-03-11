# ImageResizer

**ImageResizer** is a crafter that resizes the image to the given size.
The **ImageResizer** executor needs below parameters:

| `param_name`  | `param_remarks` |
| ------------- | ------------- |
| `target_size`  |Desired output size|
| `how`  |The interpolation method i.e. `NEAREST`, `BILINEAR`, `BICUBIC`, and `LANCZOS`|
| `channel_axis`  |The axis id of the color channel. The **-1** is the color channel info at the last axis.|


## Usage

Users can use Pod images in several ways:

1. Run with Docker (`docker run`)
   ```bash
    docker run jinahub/pod.crafter.imageresizer:0.0.15-1.0.7 --port-in 55555 --port-out 55556
    ```
    
2. Run with Flow API
   ```python
    from jina.flow import Flow

    f = (Flow()
        .add(name='my_crafter', uses='docker://jinahub/pod.crafter.imageresizer:0.0.15-1.0.7', port_in=55555, port_out=55556))
    ```
    
3. Run with Jina CLI
   ```bash
    jina pod --uses docker://jinahub/pod.crafter.imageresizer:0.0.15-1.0.7 --port-out 55556
    ```
    
4. Conventional local usage with `uses` argument
    ```bash
    jina pod --uses hub/example/imageresizer.yml --port-in 55555 --port-out 55556
    ```
    
5. Docker command

   Specify the image name along with the version tag. The snippet below uses Jina version `1.0.7`

   ```bash
    docker pull jinahub/pod.crafter.imageresizer:0.0.15-1.0.7
    ```
