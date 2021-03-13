# TikaExtractor

Based on Apache Tika, extracts text "from over a thousand different file types (such as PPT, XLS, and PDF)".
All of these file types can be parsed through a single interface, making Tika useful for search engine indexing, content analysis, translation, and much more. 

It accepts file URI and returns text of that file detected by Apache Tika. For more information, refer [Apache Tika](https://tika.apache.org/) documentation.

Following environment configuration is supported:

```bash
TIKA_OCR_STRATEGY (default: ocr_only)
TIKA_EXTRACT_INLINE_IMAGES (default: true)
TIKA_OCR_LANGUAGE (default: eng)
TIKA_TIMEOUT (default: 600)
```

## Snippets:

Initialise TikaExtractor:

| `param_name`                  | `param_remarks`                                               |
| ----------------------------- | ------------------------------------------------------------- |
| `tika_ocr_strategy`           |Type of ocr strategy, one of - no_ocr, ocr_only, ocr_and_text  |
| `tika_extract_inline_images`  |Extract inline images or not                                   |
| `tika_ocr_language`           |The language model. English by default                         |
| `tika_request_timeout`        |Timeout for server request                                     |

**NOTE**: 

- `MODULE_VERSION` is the version of the TikaExtractor, in semver format. E.g. `0.0.13`.
- `JINA_VERSION` is the version of the Jina core version with which the Docker image was built. E.g. `1.0.1` 

- Flow API

  ```python
    from jina.flow import Flow
    f = (Flow()
        .add(name='my-crafter', uses='docker://jinahub/pod.crafter.tikaextractor:MODULE_VERSION-JINA_VERSION')
    ```
- Flow YAML file
  This is the only way to provide arguments to its parameters:
  
  ```yaml
  pods:
    - name: ngt
      uses: crafters/nlp/TikaExtractor/config.yml
  ```
  
  and then in `tika.yml`:
  ```yaml
  !TikaExtractor
  metas:
    - py_modules:
        - __init__.py
  ```
- Jina CLI
  
  ```bash
  jina pod --uses docker://jinahub/pod.crafter.tikaextractor:MODULE_VERSION-JINA_VERSION
  ```
- Conventional local usage with `uses` argument
  
  ```bash
  jina pod --uses crafters/nlp/TikaExtractor/config.yml --port-in 55555 --port-out 55556
  ```
- Run with Docker (`docker run`)
 
  Specify the image name along with the version tag. The snippet below uses Jina version as `JINA_VERSION`.
  ```bash
    docker run --network host docker://jinahub/pod.crafter.tikaextractor:MODULE_VERSION-JINA_VERSION --port-in 55555 --port-out 55556
    ```