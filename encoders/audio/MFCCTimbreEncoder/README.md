# MFCCTimbreEncoder

`MFCCTimbreEncoder` is based on Mel-Frequency Cepstral Coefficients (MFCCs) which represent timbral features.
It extracts a `n_mfcc`-dimensional feature vector for each MFCC frame.
    
`MFCCTimbreEncoder` encodes an audio signal from a `Batch x Signal Length` ndarray into a
    `Batch x Concatenated Features` ndarray.


## Snippets:

The **MFCCTimbreEncoder** executor needs the following parameters:
Initialise MFCCTimbreEncoder:

| `param_name`  | `param_remarks` |
| ------------- | ------------- |
| `input_sample_rate`  | input sampling rate in Hz (22050 by default) |
| `hop_length`  | the number of samples between successive MFCC frames (512 by default) |
| `n_mfcc`      | the number of coefficients (20 by default) |
| `n_fft`       | length of the FFT window (2048 by default) |

**NOTE**: 

- `MODULE_VERSION` is the version of the MFCCTimbreEncoder, in semver format. E.g. `0.0.9`.
- `JINA_VERSION` is the version of the Jina core version with which the Docker image was built. E.g. `1.0.9` 

- Flow API

  ```python
    from jina.flow import Flow
    f = (Flow().add(name='mfcc-encoder',
                    uses='docker://jinahub/pod.encoder.mfcctimbreencoder:MODULE_VERSION-JINA_VERSION')
    ```
- Flow YAML file
  This is the only way to provide arguments to its parameters:
  
  ```yaml
  pods:
    - name: ngt
      uses: encoders/audio/MFCCTimbreEncoder/config.yml
  ```
  
  and then in `mfcc_encoder.yml`:
  ```yaml
  !MFCCTimbreEncoder
  metas:
    - py_modules:
        - __init__.py
  ```
- Jina CLI
  
  ```bash
  jina pod --uses docker://jinahub/pod.encoder.mfcctimbreencoder:MODULE_VERSION-JINA_VERSION
  ```
- Conventional local usage with `uses` argument
  
  ```bash
  jina pod --uses encoders/audio/MFCCTimbreEncoder/config.yml --port-in 55555 --port-out 55556
  ```
- Run with Docker (`docker run`)
 
  Specify the image name along with the version tag. The snippet below uses Jina version as `JINA_VERSION`.
  ```bash
    docker run --network host docker://jinahub/pod.encoder.mfcctimbreencoder:MODULE_VERSION-JINA_VERSION --port-in 55555 --port-out 55556
    ```
