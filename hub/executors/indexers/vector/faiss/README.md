# Vector Indexer based on Faiss

This image is used for running a pod with a vector indexer running inside. The vector indexer is based on Faiss. 

Please find more information about Faiss at [https://github.com/facebookresearch/faiss/](https://github.com/facebookresearch/faiss/).

## Usages

By default, the pod will save the index at `/workspace`. To run the pods in production, you will need to map a local volume to the index folder when running the pod. With the following lines, we run the pod with the local folder `./workspace ` being mapped to `/workspace`,

```bash
jina pod --image jinaai/hub.executors.indexers.vector.faiss:latest --volumes "$(pwd)/workspace"
```

To further customize the pod, you can use your own `.yml` file as well.

```bash
jina pod --image jinaai/hub.executors.indexers.vector.faiss:latest --volumes "$(pwd)/workspace --volumes "$(pwd)/yaml --yaml_path /yaml/my_awesome.yml"
```


## Build locally

You are encouraged to customize and improve the image. The following commands are for building and testing your own image,

```bash
cd hub/executors/indexers/vector/faiss
docker build -t jinaai/hub.executors.indexers.vector.your_awesome_faiss .
docker run -v "$(pwd)/workspace:/workspace" jinaai/hub.executors.indexers.vector.your_awesome_faiss:latest
```

