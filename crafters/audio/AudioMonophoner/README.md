# AudioMonophoner

**AudioMonophoner** makes the audio signal monophonic on doc-level.

The supported sound formats are:
WAV, MP3, OGG, AU,FLAC, RAW, AIFF, AIFF-C, PAF, SVX, NIST, VOC, IRCAM, W64, MAT4, MAT5, PVF, XI, HTK, SDS, AVR, WAVEX, SD2, CAF, WVE, MPC2K, RF64.

## Usage

The **AudioMonophoner** executor uses the [`librosa`](https://github.com/librosa/librosa)  library to convert the input `blob` (`ndarray` representation of the audio signal) to a dictionary of `monophonic` audio signal. 

We use Pod images in several ways:
   
1. Run with Flow API
   ```python
    from jina.flow import Flow

    f = (Flow()
        .add(name='my_encoder', uses='docker://jinahub/pod.crafter.audiomonophoner:0.0.9-1.0.1', port_in=55555, port_out=55556))
    ```
 
 2. Flow YAML file
  This is the only way to provide arguments to its parameters:

  ```yaml
  pods:
    - name: audiomonophoner
      uses: crafters/audio/AudioMonophoner/config.yml
  ```

  and then in `audiomonophoner.yml`:
  ```yaml
  !AudioMonophoner
  with:
    hostname: yourdomain.com
    port: 6379
    db: 0
  ```

3. Run with Jina CLI
   ```bash
    jina pod --uses docker://jinahub/pod.crafter.audiomonophoner:0.0.9-1.0.1 --port-in 55555 --port-out 55556
    ```
    
   Run with conventional local usage. Needs `uses` argument:
   ```bash
    jina pod --uses crafters/audio/AudioMonophoner/config.yml --port-in 55555 --port-out 55556
   ```
   
4. Run with Docker: `docker run`
   ```bash
    docker run --rm -p 55555:55555 -p 55556:55556 jinahub/pod.crafter.audiomonophoner:0.0.9-1.0.1 --port-in 55555 --port-out 55556
    ```
 
