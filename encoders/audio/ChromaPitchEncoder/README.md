# ChromaPitchEncoder

`ChromaPitchEncoder` segments  audio signal into short chroma frames. It is based on chroma spectrograms (chromagrams) which represent melodic/harmonic features.
`ChromaPitchEncoder` encodes an audio signal from a `Batch x Signal Length` ndarray into a `Batch x Concatenated Features` ndarray.

## Snippets:

The **ChromaPitchEncoder** executor needs the following parameters:
Initialise ChromaPitchEncoder:

| `param_name`  | `param_remarks` |
| ------------- | ------------- |
| `input_sample_rate`  | input sampling rate in Hz (22050 by default) |
| `hop_length`  | the number of samples between successive chroma frames (512 by default) |


**NOTE**: 

- `MODULE_VERSION` is the version of the ChromaPitchEncoder, in semver format. E.g. `0.0.9`.
- `JINA_VERSION` is the version of the Jina core version with which the Docker image was built. E.g. `1.0.2` 

- Flow API

  ```python
    from jina.flow import Flow
    f = (Flow()
        .add(name='chroma-encoder', uses='docker://jinahub/pod.encoder.chromapitchencoder:MODULE_VERSION-JINA_VERSION')
    ```
- Flow YAML file
  This is the only way to provide arguments to its parameters:
  
  ```yaml
  pods:
    - name: ngt
      uses: encoders/audio/ChromaPitchEncoder/config.yml
  ```
  
  and then in `chroma_pitch_encoder.yml`:
  ```yaml
  !ChromaPitchEncoder
  metas:
    - py_modules:
        - __init__.py
  ```
- Jina CLI
  
  ```bash
  jina pod --uses docker://jinahub/pod.encoder.chromapitchencoder:MODULE_VERSION-JINA_VERSION
  ```
- Conventional local usage with `uses` argument
  
  ```bash
  jina pod --uses encoders/audio/ChromaPitchEncoder/config.yml --port-in 55555 --port-out 55556
  ```
- Run with Docker (`docker run`)
 
  Specify the image name along with the version tag. The snippet below uses Jina version as `JINA_VERSION`.
  ```bash
    docker run --network host docker://jinahub/pod.encoder.chromapitchencoder:MODULE_VERSION-JINA_VERSION --port-in 55555 --port-out 55556
    ```