# PDFPlumberSegmenter

PDFPlumberSegmenter is a segmenter used for extracting images and text as chunks from PDF data. It stores each images and text of each page as chunks separately.

## Usage

We use Pod images in several ways:

1. Run with Docker: `docker run`
   ```bash
    docker run jinahub/pod.segmenter.pdfplumbersegmenter:0.0.1-1.1.5 --port-in 55555 --port-out 55556
    ```
    
2. Run with Flow API
   ```python
    from jina.flow import Flow

    f = (Flow()
        .add(name='my_segmenter', uses='docker://jinahub/pod.segmenter.pdfplumbersegmenter:0.0.1-1.0.1', port_in=55555, port_out=55556))
    ```
    
3. Run with Jina CLI
   ```bash
    jina pod --uses docker://jinahub/pod.segmenter.pdfplumbersegmenter:0.0.1-1.0.1 --port-in 55555 --port-out 55556
    ```
    
4. Docker command

   Specify the image name along with the version tag. In this example we use Jina version `1.0.1`

   ```bash
    docker pull jinahub/pod.segmenter.pdfplumbersegmenter:0.0.8-1.0.1
    ```
