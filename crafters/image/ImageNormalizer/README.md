# ImageNormalizer

**ImageNormalizer** is a crafter that normalizes an input image. It receives values of file names on the document-level and returns image matrix on the subdocument-level 

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
        .add(name='my-encoder', image='jinahub/pod.crafter.imagenormalizer:0.0.13-1.0.1', port_in=55555, port_out=55556))
    ```
    
3. Run with Jina CLI
   ```bash
    jina pod --uses jinahub/pod.crafter.imagenormalizer:0.0.13-1.0.1 --port-in 55555 --port-out 55556
    ```
    
4. Conventional local usage with `uses` argument
    ```bash
    jina pod --uses hub/example/imagenormalizer.yml --port-in 55555 --port-out 55556
    ```
    
5. Docker command

   Specify the image name along with the version tag. The snippet below uses Jina version `1.0.1`

   ```bash
    docker pull jinahub/pod.crafter.imagenormalizer:0.0.13-1.0.1
    ```
   
 Note:
 
 One of the limitations with the Hub Executors currently is the tags - all Executor images should have the versions appended in the name i.e.
 if the version is `0.0.8-1.0.0`, the image name would be `jinahub/pod.crafter.imagenormalizer:0.0.13-1.0.0` instead of `pod.crafter.imagenormalizer:0.0.13-1.0.1` as in the example.