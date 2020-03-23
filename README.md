# jina-hub

Jina Hub is a centralized place to host immutable Jina components, flows and applications via container images. It enables users to employ, ship, share and exchange their best-practice in different Jina search applications.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [Build Your Pod into a Docker Image](#build-your-pod-into-a-docker-image)
  - [Goal](#goal)
  - [Why?](#why)
  - [What to be packed?](#what-to-be-packed)
  - [Step-by-step Example](#step-by-step-example)
    - [1. Write Your Executor and Config](#1-write-your-executor-and-config)
    - [2. Write a 3-Line `Dockerfile`](#2-write-a-3-line-dockerfile)
      - [`FROM jinaai/jina:master-debian`](#from-jinaaijinamaster-debian)
      - [`ADD *.py mwu_encoder.yml ./`](#add-py-mwu_encoderyml-)
      - [`ENTRYPOINT ["jina", "pod", "--yaml_path", "mwu_encoder.yml"]`](#entrypoint-jina-pod---yaml_path-mwu_encoderyml)
    - [3. Build the Pod Image](#3-build-the-pod-image)
- [Use Your Pod Image](#use-your-pod-image)
  - [Use the Pod image with Docker CLI](#use-the-pod-image-with-docker-cli)
  - [Use the Pod image with Jina CLI](#use-the-pod-image-with-jina-cli)
  - [Use the Pod image in the Flow API](#use-the-pod-image-in-the-flow-api)
- [Upload Your Pod Image to Jina Hub](#upload-your-pod-image-to-jina-hub)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Build Your Pod into a Docker Image

### Goal

Instead of
```bash
jina pod --yaml_path hub/example/mwu_encoder.yml --port_in 55555 --port_out 55556
```

After this tutorial, you can use the Pod image via:
```bash
docker run jinaai/hub.example.mwu_encoder --port_in 55555 --port_out 55556
```

or, use the Pod image in the Flow API:
```python
from jina.flow import Flow

f = (Flow()
        .add(name='my-encoder', image='jinaai/hub.example.mwu_encoder', port_in=55555, port_out=55556)
        .add(name='my-indexer', yaml_path='indexer.yml'))
```

or, use the Pod image via Jina CLI
```bash
jina pod --image jinaai/hub.example.mwu_encoder --port_in 55555 --port_out 55556
```

More information about [the usage can be found here](#use-the-pod-image).


### Why?

So you have implemented your awesome executor and want to reuse it in another Jina application, or share it with people in the world. Kind as you are, you want to offer people a ready-to-use interface and the best-practice without hassling them to repeat all steps and pitfalls you have done. The best way is thus to pack everything (python file, YAML config, pre-trained data, dependencies) into a container image and use Jina as the entry point. You can also annotate your image with some meta information to facilitate the search, archive and classification.

Here are a list of reasons that may motivate you to build a Pod image:

- You want to use one of the built-in executor (e.g. pytorch-based) and you don't want to install pytorch dependencies on the host.
- You modify or write a new executor and want to reuse it in another project, without touching [`jina-ai/jina`](https://github.com/jina-ai/jina/).
- You customize the driver and want to reuse it in another project, without touching [`jina-ai/jina`](https://github.com/jina-ai/jina/).
- You have a self-built library optimized for your architecture (e.g. tensorflow/numpy on GPU/CPU/x64/arm64), and you want this specific Pod to benefit from it.
- Your awesome executor requires certain Linux headers that can only be installed via `apt` or `yum`, but you don't have `sudo` on the host.
- You executor relies on a pretrained model, you want to include this 100MB file into the image so that people don't need to download it again.  
- You use Kubernetes or Docker Swarm and this orchestration framework requires each microservice to run as a Docker container.
- You are using Jina on the cloud and you want to deploy a immutable Pod and version control it.
- You have figured out a set of parameters that works best for an executor, you want to write it down in a YAML config and share it to others.
- You are debugging, doing try-and-error on exploring new packages, and you don't want ruin your local dev environments. 


### What to be packed?

Typically, the following files are packed into the Docker image:

- `Dockerfile`: describes the dependency setup and expose the entry point;
- `*.py`: the Python file(s) describes the executor logic, if applicable;
- `*.yml`: a YAML file describes the executor arguments and configs, if you want users to use your config;
- Other data files that may be required to run the executor, e.g. pretrained model, fine-tuned model, home-made data.

Except `Dockerfile`, all others are optional to build a valid Pod image depending on your case. 
    
### Step-by-step Example

In this example, we consider the scenario where we creates a new executor and want to reuse it in another project, without touching [`jina-ai/jina`](https://github.com/jina-ai/jina/). All files required in this guide is available at [`hub/example/`](/hub/example).

#### 1. Write Your Executor and Config

We write a new dummy encoder named `MWUEncoder` in [`mwu_encoder.py`](hub/example/mwu_encoder.py) which encodes any input into a random 3-dimensional vector. This encoder has a dummy parameter `greetings` which prints a greeting message on start and on every encode. In [`mwu_encoder.yml`](hub/example/mwu_encoder.yml), the `metas.py_modules` tells Jina to load the class of this executor from `mwu_encoder.py`.

```yaml
!MWUEncoder
with:
  greetings: im from internal yaml!
metas:
  name: my-mwu-encoder
  py_modules: mwu_encoder.py
  workspace: ./
```

The documentations of the YAML syntax [can be found at here](http://0.0.0.0:8000/chapters/yaml/yaml.html#executor-yaml-syntax). 

#### 2. Write a 3-Line `Dockerfile`

The `Dockerfile` in this example is as simple as three lines, 

```Dockerfile
FROM jinaai/jina:master-debian

ADD *.py mwu_encoder.yml ./

ENTRYPOINT ["jina", "pod", "--yaml_path", "mwu_encoder.yml"]
```

##### `FROM jinaai/jina:master-debian` 
In the first line, we choose `jinaai/jina:master-debian` as the base image, which corresponds to the latest master of [`jina-ai/jina`](https://github.com/jina-ai/jina). But of course you are free to use others, e.g. `tensorflow/tensorflow:nightly-gpu-jupyter`. In practice, whether or not using Jina base image on the dependencies you would like to introduce. For example, someone provides a hard-to-compile package as a Docker image, much harder than compiling/installing Jina itself. In this case, you may want to use this image as the base image to save some troubles. But don't forget to install python 3.7 and Jina afterwards, e.g.

```Dockerfile
FROM awesome-gpu-optimized-kit

RUN pip install jina --no-cache-dir --compile
```

The ways of [installing Jina can be at found here](https://github.com/jina-ai/jina#run-without-docker).

In this example, our dummy `MWUEncoder` only requires Jina and does not need any third-party framework. Thus, `jinaai/jina:master-debian` is used.

##### `ADD *.py mwu_encoder.yml ./`

The second step is to add *all* necessary files to the image. Typically, Python codes, YAML config and some data files.

In this example, our dummy `MWUEncoder` does not require extra data files.

##### `ENTRYPOINT ["jina", "pod", "--yaml_path", "mwu_encoder.yml"]` 

The last step is to specify the entrypoint of this image, usually via `jina pod`.

In this example, we set `mwu_encoder.yml` as a default YAML config. So if the user later run

```bash
docker run jinaai/hub.example.mwu_encoder
```
 
It is equal to:
```bash
jina pod --yaml_path hub/example/mwu_encoder.yml
```

Any followed key-value arguments after `docker run jinaai/hub.example.mwu_encoder` will be passed to `jina pod`. For example,

```bash
docker run jinaai/hub.example.mwu_encoder --port_in 55555 --port_out 55556
```
 
It is equal to:
```bash
jina pod --yaml_path hub/example/mwu_encoder.yml --port_in 55555 --port_out 55556
```

One can also override the internal YAML config by giving an out-of-docker external YAML config via:

```bash
docker run $(pwd)/hub/example/mwu_encoder_ext.yml:/ext.yml jinaai/hub.example.mwu_encoder --yaml_path /ext.yml
```


#### 3. Build the Pod Image

Now you can build the Pod image via `docker build`:

```bash
cd hub/example
docker build -t jinaai/hub.example.mwu_encoder .
```

Depending on whether you want to use the latest Jina image, you may first pull it via `docker pull jinaai/jina:master-debian` before the build. For the sake of immutability, `docker build` will not automatically pull the latest image for you.

Congratulations! You can now re-use this Pod image how ever you want.

## Use Your Pod Image

### Use the Pod image with Docker CLI

The most powerful way to use this Pod image is via Docker CLI directly:

```bash
docker run --rm -p 55555:55555 -p 55556:55556 jinaai/hub.example.mwu_encoder --port_in 55555 --port_out 55556
```

Note, the exposure of ports `-p 55555:55555 -p 55556:55556` is required for other Pods (local/remote) to communicate this Pod. One may also want to use `--network host` and let the Pod share the network layer of the host.
 
All parameters supported by `jina pod --help` can be followed after `docker run jinaai/hub.example.mwu_encoder`.

One can mount a host path to the container via `--volumes` or `-v`. For example, to override the internal YAML config, one can do

```bash
# assuming $pwd is the root dir of this repo 
docker run --rm -v $(pwd)/hub/example/mwu_encoder_ext.yml:/ext.yml jinaai/hub.example.mwu_encoder --yaml_path /ext.yml
```

```text
MWUEncoder@ 1[S]:look at me! im from an external yaml!
MWUEncoder@ 1[S]:initialize MWUEncoder from a yaml config
 BasePea-0@ 1[I]:setting up sockets...
 BasePea-0@ 1[I]:input tcp://0.0.0.0:36109 (PULL_BIND) 	 output tcp://0.0.0.0:58191 (PUSH_BIND)	 control over tcp://0.0.0.0:52365 (PAIR_BIND)
 BasePea-0@ 1[S]:ready and listening
```

To override the predefined entrypoint via `--entrypoint`, e.g.

```bash
docker run --rm --entrypoint "jina" jinaai/hub.example.mwu_encoder check
```

### Use the Pod image with Jina CLI

Another way to use the Pod image is simply give it to `jina pod` via `--image`,
```bash
jina pod --image jinaai/hub.example.mwu_encoder
```

```text
🐳 MWUEncoder@ 1[S]:look at me! im from internal yaml!
🐳 MWUEncoder@ 1[S]:initialize MWUEncoder from a yaml config
🐳 BasePea-0@ 1[I]:setting up sockets...
🐳 BasePea-0@ 1[I]:input tcp://0.0.0.0:59608 (PULL_BIND) 	 output tcp://0.0.0.0:59609 (PUSH_BIND)	 control over tcp://0.0.0.0:59610 (PAIR_BIND)
ContainerP@69041[S]:ready and listening
🐳 BasePea-0@ 1[S]:ready and listening
```

Note the 🐳 represents that the log is piping from a Docker container.

See `jina pod --help` for more usage.

### Use the Pod image in the Flow API

Finally, one can use it via Flow API as well, e.g.

```python
from jina.flow import Flow

f = (Flow()
        .add(name='my-encoder', image='jinaai/hub.example.mwu_encoder',
             volumes='./abc', yaml_path='hub/example/mwu-encoder/mwu_encoder_ext.yml', 
             port_in=55555, port_out=55556)
        .add(name='my-indexer', yaml_path='indexer.yml'))
```

## Upload Your Pod Image to Jina Hub

TBA 

