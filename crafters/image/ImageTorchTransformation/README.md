# ImageTorchTransformation

Apply torchvision transforms on image batches.

This crafter creates a `Compose()` transform using the list of transforms provided.
The Numpy input is converted to tensor before applying the transform. Hence `ToTensor()` transform is not required.

Note: 
- The input images are required to be of the same dimensions to enable creating a batch of shape, `[B, H, W, C]`
- It is not recommended to use inherently random transforms and, the transforms should support tensor inputs

## Usage
Users can use Pod images in several ways:


1. With Flow API
   ```python
    from jina.flow import Flow
    f = (Flow().add(name='my_crafter', 
                    uses='docker://jinahub/pod.crafter.imagetorchtransformation:0.0.1-1.1.9',port_in=55555, port_out=55556))
    ```

2. With Flow YAML file
    This is the only way to provide arguments to its parameters:

    ```yaml
    pods:
    - name: imagetorchtransformation
        uses: crafters/image/ImageTorchTransformation/config.yml
    ```

    and then in `imagetorchtransformation.yml`:
    ```yaml
    !ImageTorchTransformation
    with:
    hostname: yourdomain.com
    port: 6379
    db: 0
    ```


3. With Jina CLI
    ```bash
    jina pod --uses docker://jinahub/pod.crafter.imagetorchtransformation:0.0.1-1.1.9 --port-in=55555 --port-out 55556
    ```

    Conventional local usage with `uses` argument
    ```bash
    jina pod --uses crafters/image/ImageTorchTransformation/config.yml --port-in 55555 --port-out 55556
    ```

 4. With Docker (`docker run`)
    ```bash
    docker run --rm -p 55555:55555 -p 55556:55556 jinahub/pod.crafter.imagetorchtransformation:0.0.1-1.1.9 --port-in 55555 --port-out 55556
    ```

## Example:

    ```
    from jina.hub.crafters.image.ImageTorchTransformation
    import ImageTorchTransformation
    import numpy
    import cv2
    
    sample = cv2.imread(sample.png"))[:, :, ::-1]
    batch_size = 5
    image_batch = numpy.stack([sample]*batch_size)

    transforms = [
        {"CenterCrop": dict(size=(300))},
        {"Resize": dict(size=(244, 244))},
        "RandomVerticalFlip",
        {"Normalize": dict(mean=(0.485, 0.456, 0.406), 
                            std=(0.229, 0.224, 0.225))
        },
    ]

    crafter = ImageTorchTransformation(transforms)
    crafted_imgs = crafter.craft(image_batch)
    ```
