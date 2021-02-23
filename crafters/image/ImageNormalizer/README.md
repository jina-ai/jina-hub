# ImageNormalizer

**ImageNormalizer** is a crafter that normalizes an input image.

The **ImageNormalizer** executor needs the following parameters:

| `param_name`  | `param_remarks` |
| ------------- | ------------- |
| `target_size`  |It's the desired output size.  |
| `img_mean`  |It's the mean of the images in `RGB` channels.  |
| `img_std`  |It's the std of the images in `RGB` channels.  |
| `resize_dim`  |It's the desired size that images will be resized to. They are resized before the cropping |
| `channel_axis`  |It's the axis id of the color channel.  |

## Usage

Users can use Pod images in several ways:

1. Run with Docker (`docker run`)
   ```bash
    docker run jinahub/pod.crafter.imagenormalizer:0.0.13-1.0.1 --port-in 55555 --port-out 55556
    ```
    
2. Run with Flow API
   ```python
    from jina.flow import Flow

    f = (Flow()
        .add(name='my_crafter', uses='docker://jinahub/pod.crafter.imagenormalizer:0.0.13-1.0.1'))
    ```
    
3. Run with Jina CLI
   ```bash
    jina pod --uses docker://jinahub/pod.crafter.imagenormalizer:0.0.13-1.0.1 --port-in 55555 --port-out 55556
    ```
    
4. Conventional local usage with `uses` argument
    ```bash
    jina pod --uses hub/example/imagenormalizer.yml --port-in 55555 --port-out 55556
    ```
    
5. Docker command to download the image

   Specify the image name along with the version tag. The snippet below uses Jina version `1.0.1`

   ```bash
    docker pull jinahub/pod.crafter.imagenormalizer:0.0.13-1.0.1
    ```