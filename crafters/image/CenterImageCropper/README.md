# CenterImageCropper

CenterImageCropper crops the image with the center crop box. The coordinate is the same coordinate-system.

## Snippets:

Initialise CenterImageCropper:

`SptagIndexer(method, samples, tpt_number, tpt_leaf_size, neighborhood_size, graph_neighborhood_size, cef, max_check_for_refined_graph, num_threads, max_ckeck, dist_calc_method, bkt_number, bkt_meansk, kdt_number)`
| File                            | Descriptions     |
|---------------------------------|------------------|
| `target_size`                   | Desired output   |
| `channel_axis`                  | Axis for channel |

**NOTE**: 

- `MODULE_VERSION` is the version of the CenterImageCropper, in semver format. E.g. `0.0.13`.
- `JINA_VERSION` is the version of the Jina core version with which the Docker image was built. E.g. `1.0.8` 

- Flow API

  ```python
    from jina.flow import Flow
    f = (Flow()
        .add(name='my-indexer', uses='docker://jinahub/pod.crafter.centerimagecropper:MODULE_VERSION-JINA_VERSION')
    ```
- Flow YAML file
  This is the only way to provide arguments to its parameters:
  
  ```yaml
  pods:
    - name: ngt
      uses: indexers/vector/CenterImageCropper/config.yml
  ```
  
  and then in `imagecropper.yml`:
  ```yaml
  !CenterImageCropper
  with:
    hostname: yourdomain.com
    port: 6379
    db: 0
  ```
- Jina CLI
  
  ```bash
  jina pod --uses docker://jinahub/pod.crafter.centerimagecropper:MODULE_VERSION-JINA_VERSION
  ```
- Conventional local usage with `uses` argument
  
  ```bash
  jina pod --uses indexers/vector/CenterImageCropper/config.yml --port-in 55555 --port-out 55556
  ```
- Run with Docker (`docker run`)
 
  Specify the image name along with the version tag. The snippet below uses Jina version as `JINA_VERSION`.
  ```bash
    docker run --network host docker://jinahub/pod.crafter.centerimagecropper:MODULE_VERSION-JINA_VERSION --port-in 55555 --port-out 55556
    ```