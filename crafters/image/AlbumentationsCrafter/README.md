# AlbumentationsCrafter

This crafter provides access to any of the transforms from the [Albumentations](https://github.com/albumentations-team/albumentations/) package. It also allows you to chain any number of transforms in one crafter using ``A.Compose``.

> Note that Albumentation's transform are meant to be random, which is usually not what you want in indexing/search context. The crafter automatically adds `always_apply=True` to all transforms, but you should watch out for other sources of "randomness" (see AlbumentationsCrafter documentation for more detail)

## Usage

Users can use Pod images in several ways:

1. Run with Flow API
   ```python
    from jina.flow import Flow
    f = (Flow()
        .add(name='my_crafter', uses='docker://jinahub/pod.crafter.albumentationscrafter:0.0.3-1.0.7', port_in=55555, port_out=55556))
    ```

2. Flow YAML file
  This is the only way to provide arguments to its parameters:
  
  ```yaml
  pods:
    - name: albumentationscrafter
      uses: crafters/image/AlbumentationsCrafter/config.yml
  ```
  
  and then in `albumentation.yml`:
  ```yaml
  !AlbumentationsCrafter
  with:
    hostname: yourdomain.com
    port: 6379
    db: 0
  ```


3. Run with Jina CLI
   ```bash
    jina pod --uses docker://jinahub/pod.crafter.albumentationscrafter:0.0.3-1.0.7 --port-in=55555 --port-out 55556
    ```

   Conventional local usage with `uses` argument
    ```bash
    jina pod --uses crafters/image/AlbumentationsCrafter/config.yml --port-in 55555 --port-out 55556
    ```
    
 4. Run with Docker (`docker run`)
   ```bash
    docker run --rm -p 55555:55555 -p 55556:55556 jinahub/pod.crafter.albumentationscrafter:0.0.3-1.0.7 --port-in 55555 --port-out 55556
    ```


## Simple example

Here's a simple example of how to use this crafter. We'll be transforming this image:

| `rubi.png` |
|:--:|
| ![alt text](tests/rubi.png "rubi") |

We first read in the image

``` python
import numpy as np
import PIL

img = np.asarray(PIL.Image.open('rubi.png'))
```

First, let's apply the equivalent of `A.VerticalFlip`. To do that, we can define the crafter and transform the image like so

``` python
crafter = AlbumentationsCrafter(['VerticalFlip'])
flipped_img = crafter.craft('rubi.png')
```

The result should look like this

| `rubi_flip.png` |
|:--:|
| ![alt text](tests/rubi_flip.png "rubi") |

Now let's do something more complicated - crop the center of the image. We are aiming for the equivalent of `A.CenterCrop(height=100, width=100)`. We can achieve this with

``` python
transform = {'CenterCrop': dict(height=100, width=100)}
crafter = AlbumentationsCrafter([transform])
center_crop_img = crafter.craft('rubi.png')
```

| `rubi_center_crop.png` |
|:--:|
| ![alt text](tests/rubi_center_crop.png "rubi") |

Here's a good time to show how the yaml definition of such a transformer would look like:

``` yaml
!AlbumentationsCrafter
with:
  transforms:
    - CenterCrop:
        width: 100
        height: 100
```

## Composing transforms

Composing transforms is straightforward. We'll show how to achieve the equivalent of

``` python
A.Compose([
    A.VerticalFlip(),
    A.CenterCrop(height=100, width=100)
])
```

Here's how to define the crafter in python

``` python
transforms = [
    'VerticalFlip',
    {'CenterCrop': dict(height=100, width=100)}
]
crafter = AlbumentationsCrafter(transforms)
```

And here's how to do it in yaml

``` yaml
!AlbumentationsCrafter
with:
  transforms:
    - VerticalFlip
    - CenterCrop:
        width: 100
        height: 100
```
