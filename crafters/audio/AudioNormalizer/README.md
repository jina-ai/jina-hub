# AudioNormalizer

**AudioNormalizer** is a crafter that takes signal audio as input and normalizes it using the [Librosa](https://librosa.org/doc/latest/index.html) library. This is done on the Document-level. 

The **AudioNormalizer** executor needs only one parameter:

| `param_name`  | `param_remarks` |
| ------------- | ------------- |
| `blob`  |It's the desired audio signal to be normalized.  |

## Usage

Users can use Pod images in several ways:

1. Run with Docker (`docker run`)
   ```bash
    docker run jinahub/pod.crafter.audionormalizer:0.0.9-1.0.1 --port-in 55555 --port-out 55556
    ```
    
2. Run with Flow API
   ```python
    from jina.flow import Flow

    f = (Flow()
        .add(name='my-encoder', image='jinahub/pod.crafter.audionormalizer:0.0.9-1.0.1', port_in=55555, port_out=55556))
    ```
    
3. Run with Jina CLI
   ```bash
    jina pod --uses jinahub/pod.crafter.audionormalizer:0.0.9-1.0.1 --port-in 55555 --port-out 55556
    ```
    
4. Conventional local usage with `uses` argument
    ```bash
    jina pod --uses hub/example/audionormalizer.yml --port-in 55555 --port-out 55556
    ```
    
5. Docker command

   Specify the image name along with the version tag. The snippet below uses Jina version `1.0.1`

   ```bash
    docker pull jinahub/pod.crafter.audionormalizer:0.0.9-1.0.1
    ```
   
 Note:
 
 One of the limitations with the Hub Executors currently is the tags - all Executor images should have the versions appended in the name i.e.
 if the version is `0.0.8-1.0.0`, the image name would be `jinahub/pod.crafter.audionormalizer:0.0.9-1.0.0` instead of `pod.crafter.audionormalizer:0.0.9-1.0.1` as in the example.
