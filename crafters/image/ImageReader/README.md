# ImageReader

**ImageReader** is a crafter that loads an image, either from a given file path or directly from bytes and save the `ndarray` of the image in the Document. It reads the image specified in `buffer` and save the `ndarray` of the image in the `blob` of the document.

The **ImageReader** executor needs only one parameter:

| `param_name`  | `param_remarks` |
| ------------- | ------------- |
| `channel_axis`  |The axis id of the color channel. The **-1** is the color channel info at the last axis.|

## Usage

Users can use Pod images in several ways:

1. Run with Docker (`docker run`)
   ```bash
    docker run jinahub/pod.crafter.imagereader:0.0.13-1.0.1 --port-in 55555 --port-out 55556
    ```
    
2. Run with Flow API
   ```python
    from jina.flow import Flow

    f = (Flow()
        .add(name='my_encoder', uses='docker://jinahub/pod.crafter.imagereader:0.0.13-1.0.1', port_in=55555, port_out=55556))
    ```
    
3. Run with Jina CLI
   ```bash
    jina pod --uses docker://jinahub/pod.crafter.imagereader:0.0.13-1.0.1 --port-out 55556
    ```
    
4. Conventional local usage with `uses` argument
    ```bash
    jina pod --uses hub/example/imagereader.yml --port-in 55555 --port-out 55556
    ```
    
5. Docker command

   Specify the image name along with the version tag. The snippet below uses Jina version `1.0.1`

   ```bash
    docker pull jinahub/pod.crafter.imagereader:0.0.13-1.0.1
    ```
   
 Note:
 
 One of the limitations with the Hub Executors currently is the tags - all Executor images should have the versions appended in the name i.e.
 if the version is `0.0.8-1.0.0`, the image name would be `pod.crafter.imagereader:0.0.13-1.0.0` instead of `pod.crafter.imagereader:0.0.13-1.0.1` as in the example.
