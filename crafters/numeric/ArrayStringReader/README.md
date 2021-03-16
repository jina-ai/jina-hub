# ArrayStringReader

**ArrayStringReader** is a numeric crafter that converts a string of numbers into a numpy array and save to the Document. 


The **ArrayStringReader** executor needs the following parameters:

## Snippets:

Initialise ArrayStringReader:

| `param_name`                  | `param_remarks`                    |
| ----------------------------- | ---------------------------------- |
| `as_type`                     | The numpy array will be this type  |
| `delimiter`                   | Delimiter between numbers          |

**NOTE**: 

- `MODULE_VERSION` is the version of the ArrayStringReader, in semver format. E.g. `0.0.8`.
- `JINA_VERSION` is the version of the Jina core version with which the Docker image was built. E.g. `1.0.8` 

- Flow API

  ```python
    from jina.flow import Flow
    f = (Flow()
        .add(name='my-crafter', uses='docker://jinahub/pod.crafter.arraystringreader:MODULE_VERSION-JINA_VERSION')
    ```
- Flow YAML file
  This is the only way to provide arguments to its parameters:
  
  ```yaml
  pods:
    - name: arraystringreader
      uses: crafters/numeric/ArrayStringReader/config.yml
  ```
  
  and then in `string-reader.yml`:
  ```yaml
  !ArrayStringReader
  metas:
    - py_modules:
        - __init__.py
  ```
- Jina CLI
  
  ```bash
  jina pod --uses docker://jinahub/pod.crafter.arraystringreader:MODULE_VERSION-JINA_VERSION
  ```
- Conventional local usage with `uses` argument
  
  ```bash
  jina pod --uses crafters/numeric/ArrayStringReader/config.yml --port-in 55555 --port-out 55556
  ```
- Run with Docker (`docker run`)
 
  Specify the image name along with the version tag. The snippet below uses Jina version as `JINA_VERSION`.
  ```bash
    docker run --network host docker://jinahub/pod.crafter.arraystringreader:MODULE_VERSION-JINA_VERSION --port-in 55555 --port-out 55556
    ```