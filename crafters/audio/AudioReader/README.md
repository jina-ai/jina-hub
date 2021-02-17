# AudioReader

**AudioReader** reads and resamples audio signal on doc-level.  Loads an audio file as `ndarray` and resamples the audio signal to the target sampling rate(default 22050Hz).

The supported sound formats are:
WAV, MP3, OGG, AU,FLAC, RAW, AIFF, AIFF-C, PAF, SVX, NIST, VOC, IRCAM, W64, MAT4, MAT5, PVF, XI, HTK, SDS, AVR, WAVEX, SD2, CAF, WVE, MPC2K, RF64.

## Usage

The **AudioReader** executor needs only one parameter:

| `param_name`  | `param_remarks` |
| ------------- | ------------- |
| `target_sample_rate`  | The desired sample rate. Requires a scalar number bigger than 0  |


We use Pod images in several ways:

1. Run with Docker: `docker run`
   ```bash
    docker run jinahub/pod.crafter.audioreader:0.0.8-1.0.1 --port-in 55555 --port-out 55556
    ```
    
2. Run with Flow API
   ```python
    from jina.flow import Flow

    f = (Flow()
        .add(name='my_encoder', uses='docker://jinahub/pod.crafter.audioreader:0.0.8-1.0.1', port_in=55555, port_out=55556))
    ```
    
3. Run with Jina CLI
   ```bash
    jina pod --uses docker://jinahub/pod.crafter.audioreader:0.0.8-1.0.1 --port-in 55555 --port-out 55556
    ```
    
4. Run with conventional local usage. Needs `uses` argument:
   ```bash
    jina pod --uses hub/example/audioreader.yml --port-in 55555 --port-out 55556
   ```
    
5. Docker command

   Specify the image name along with the version tag. In this example we use Jina version `1.0.1`

   ```bash
    docker pull jinahub/pod.crafter.audioreader:0.0.8-1.0.1
    ```
   
 Note:
 
 One of the limitations with the Hub Executors currently is the tags - all Executor images should have the versions appended in the name i.e.
 if the version is `0.0.8-1.0.0`, the image name would be `jinahub/pod.crafter.audioreader:0.0.8-1.0.0` instead of `jinahub/pod.crafter.audioreader:0.0.8-1.0.1` as in the example.