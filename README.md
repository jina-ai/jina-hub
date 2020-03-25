# jina-hub

Jina Hub is a centralized place to host immutable Jina components, flows and applications via container images. It enables users to employ, ship, share and exchange their best-practice in different Jina search applications.

|                                    |         I want to ...         |                                  |
|:----------------------------------:|:-----------------------------:|:--------------------------------:|
| <h1>📦</h1><br>[build my own Pod image](#build-your-pod-into-a-docker-image) | <h1>🐳</h1><br>[use Pod image in my project](#use-your-pod-image) | <h1>🆙️</h1><br>[publish Pod image to Jina Hub](#publish-your-pod-image-to-jina-hub) |


<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [Build Your Pod into a Docker Image](#build-your-pod-into-a-docker-image)
  - [Goal](#goal)
  - [Why?](#why)
  - [What Should be in the Image?](#what-should-be-in-the-image)
  - [Step-by-Step Example](#step-by-step-example)
- [Use Your Pod Image](#use-your-pod-image)
  - [Use the Pod image via Docker CLI](#use-the-pod-image-via-docker-cli)
  - [Use the Pod image via Jina CLI](#use-the-pod-image-via-jina-cli)
  - [Use the Pod image via Flow API](#use-the-pod-image-via-flow-api)
- [Publish Your Pod Image to Jina Hub](#publish-your-pod-image-to-jina-hub)
  - [What Files Need to be Uploaded?](#what-files-need-to-be-uploaded)
  - [Schema of `manifest.yml`](#schema-of-manifestyml)
  - [Steps to Publish Your Image](#steps-to-publish-your-image)
  - [Why My Upload Fails on the CICD?](#why-my-upload-fails-on-the-cicd)
- [License](#license)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


## Build Your Pod into a Docker Image

### Goal

Instead of
```bash
jina pod --yaml_path hub/example/mwu_encoder.yml --port_in 55555 --port_out 55556
```

After this tutorial, you can use the Pod image via:
```bash
docker run jinaai/hub.examples.mwu_encoder --port_in 55555 --port_out 55556
```

or, use the Pod image in the Flow API:
```python
from jina.flow import Flow

f = (Flow()
        .add(name='my-encoder', image='jinaai/hub.examples.mwu_encoder', port_in=55555, port_out=55556)
        .add(name='my-indexer', yaml_path='indexer.yml'))
```

or, use the Pod image via Jina CLI
```bash
jina pod --image jinaai/hub.examples.mwu_encoder --port_in 55555 --port_out 55556
```

More information about [the usage can be found here](#use-your-pod-image).


### Why?

So you have implemented an awesome executor and want to reuse it in another Jina application, or share it with people in the world. Kind as you are, you want to offer people a ready-to-use interface without hassling them to repeat all steps and pitfalls you have done. The best way is thus to pack everything (python file, YAML config, pre-trained data, dependencies) into a container image and use Jina as the entry point. You can also annotate your image with some meta information to facilitate the search, archive and classification.

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


### What Should be in the Image?

Typically, the following files are packed into the container image:

| File             | Descriptions                                                                                        |
|------------------|-----------------------------------------------------------------------------------------------------|
| `Dockerfile`     | describes the dependency setup and expose the entry point;                                          |
| `build.args`     | metadata of the image, author, tags, etc. help the Hub to index and classify your image             |
| `*.py`           | describes the executor logic written in Python, if applicable;                                      |
| `*.yml`          | a YAML file describes the executor arguments and configs, if you want users to use your config;     |
| Other data files | may be required to run the executor, e.g. pre-trained model, fine-tuned model, home-made data.      |

Except `Dockerfile`, all others are optional to build a valid Pod image depending on your case. `build.args` is only required when you want to [upload your image to Jina Hub](#publish-your-pod-image-to-jina-hub).
    
### Step-by-Step Example

In this example, we consider the scenario where we creates a new executor and want to reuse it in another project, without touching [`jina-ai/jina`](https://github.com/jina-ai/jina/). All files required in this guide is available at [`hub/examples/mwu_encoder`](/hub/examples/mwu_encoder).

#### 1. Write Your Executor and Config

We write a new dummy encoder named `MWUEncoder` in [`mwu_encoder.py`](hub/examples/mwu_encoder/mwu_encoder.py) which encodes any input into a random 3-dimensional vector. This encoder has a dummy parameter `greetings` which prints a greeting message on start and on every encode. In [`mwu_encoder.yml`](hub/examples/mwu_encoder/mwu_encoder.yml), the `metas.py_modules` tells Jina to load the class of this executor from `mwu_encoder.py`.

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

Let's now look at these three lines one by one.

>
```Dockerfile
FROM jinaai/jina:master-debian
``` 

In the first line, we choose `jinaai/jina:master-debian` as [the base image](https://docs.docker.com/glossary/#base-image), which corresponds to the latest master of [`jina-ai/jina`](https://github.com/jina-ai/jina). But of course you are free to use others, e.g. `tensorflow/tensorflow:nightly-gpu-jupyter`. 

In practice, whether to use Jina base image depends on the dependencies you would like to introduce. For example, someone provides a hard-to-compile package as a Docker image, much harder than compiling/installing Jina itself. In this case, you may want to use this image as the base image to save some troubles. But don't forget to install Python >=3.7 (if not included) and Jina afterwards, e.g.

> 
```Dockerfile
FROM awesome-gpu-optimized-kit

RUN pip install jina --no-cache-dir --compile
```

The ways of [installing Jina can be at found here](https://github.com/jina-ai/jina#run-without-docker).

In this example, our dummy `MWUEncoder` only requires Jina and does not need any third-party framework. Thus, `jinaai/jina:master-debian` is used.

```Dockerfile
ADD *.py mwu_encoder.yml ./
```

The second step is to add *all* necessary files to the image. Typically, Python codes, YAML config and some data files.

In this example, our dummy `MWUEncoder` does not require extra data files.

> 
```Dockerfile
ENTRYPOINT ["jina", "pod", "--yaml_path", "mwu_encoder.yml"]
``` 

The last step is to specify the entrypoint of this image, usually via `jina pod`.

In this example, we set `mwu_encoder.yml` as a default YAML config. So if the user later run

```bash
docker run jinaai/hub.examples.mwu_encoder
```
 
It is equal to:
```bash
jina pod --yaml_path hub/example/mwu_encoder.yml
```

Any followed key-value arguments after `docker run jinaai/hub.examples.mwu_encoder` will be passed to `jina pod`. For example,

```bash
docker run jinaai/hub.examples.mwu_encoder --port_in 55555 --port_out 55556
```
 
It is equal to:
```bash
jina pod --yaml_path hub/example/mwu_encoder.yml --port_in 55555 --port_out 55556
```

One can also override the internal YAML config by giving an out-of-docker external YAML config via:

```bash
docker run $(pwd)/hub/example/mwu_encoder_ext.yml:/ext.yml jinaai/hub.examples.mwu_encoder --yaml_path /ext.yml
```


#### 3. Build the Pod Image

Now you can build the Pod image via `docker build`:

```bash
cd hub/example
docker build -t jinaai/hub.examples.mwu_encoder .
```

Depending on whether you want to use the latest Jina image, you may first pull it via `docker pull jinaai/jina:master-debian` before the build. For the sake of immutability, `docker build` will not automatically pull the latest image for you.

Congratulations! You can now re-use this Pod image how ever you want.

## Use Your Pod Image

### Use the Pod image via Docker CLI

The most powerful way to use this Pod image is via Docker CLI directly:

```bash
docker run --rm -p 55555:55555 -p 55556:55556 jinaai/hub.examples.mwu_encoder --port_in 55555 --port_out 55556
```

Note, the exposure of ports `-p 55555:55555 -p 55556:55556` is required for other Pods (local/remote) to communicate this Pod. One may also want to use `--network host` and let the Pod share the network layer of the host.
 
All parameters supported by `jina pod --help` can be followed after `docker run jinaai/hub.examples.mwu_encoder`.

One can mount a host path to the container via `--volumes` or `-v`. For example, to override the internal YAML config, one can do

```bash
# assuming $pwd is the root dir of this repo 
docker run --rm -v $(pwd)/hub/example/mwu_encoder_ext.yml:/ext.yml jinaai/hub.examples.mwu_encoder --yaml_path /ext.yml
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
docker run --rm --entrypoint "jina" jinaai/hub.examples.mwu_encoder check
```

### Use the Pod image via Jina CLI

Another way to use the Pod image is simply give it to `jina pod` via `--image`,
```bash
jina pod --image jinaai/hub.examples.mwu_encoder
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

### Use the Pod image via Flow API

Finally, one can use it via Flow API as well, e.g.

```python
from jina.flow import Flow

f = (Flow()
        .add(name='my-encoder', image='jinaai/hub.examples.mwu_encoder',
             volumes='./abc', yaml_path='hub/examples/mwu-encoder/mwu_encoder_ext.yml', 
             port_in=55555, port_out=55556)
        .add(name='my-indexer', yaml_path='indexer.yml'))
```

## Publish Your Pod Image to Jina Hub

You can contribute your executor into this repository and it will be automatically built into an image and published to the Jina Hub via our CI/CD pipeline.

### What Files Need to be Uploaded?

Typically, the following files are required:

| File             | Descriptions                                                                                        |
|------------------|-----------------------------------------------------------------------------------------------------|
| `Dockerfile`     | describes the dependency setup and expose the entry point;                                          |
| `manifest.yml`   | metadata of the image, author, tags, etc. help the Hub to index and classify your image             |
| `*.py`           | describes the executor logic written in Python, if applicable;                                      |
| `*.yml`          | a YAML file describes the executor arguments and configs, if you want users to use your config;     |

Note, large binary files (such as pretrained model, auxiliary data) are **not** recommended to upload to this repository. You can use `RUN wget ...` or `RUN curl` inside the `Dockerfile` to download it from the web during the build.

### Schema of `manifest.yml`

`manifest.yml` must exist if you want to publish your Pod image to Jina Hub.

`manifest.yml` annotates your image so that it can be managed by Jina Hub. To get better appealing on Jina Hub, you should carefully set `manifest.yml` to the correct values.

| Key | Description | Default |
| --- | --- | --- |
| `name` | Human-readable title of the image (must match with `^[a-zA-Z_$][a-zA-Z_\s\-$0-9]{3,20}$`) | Required |
| `description` | Human-readable description of the software packaged in the image | Required |
| `author` | contact details of the people or organization responsible for the image (string) | `Jina AI Dev-Team (dev-team@jina.ai)` |
| `url` | URL to find more information on the image (string) | `https://jina.ai` |
| `documentation` | URL to get documentation on the image (string) | `https://docs.jina.ai` |
| `version` | version of the image, it should be [Semantic versioning-compatible](http://semver.org/) | `0.0.0` |
| `vendor` | the name of the distributing entity, organization or individual (string) | `Jina AI Limited` |
| `license` | license under which contained software is distributed, it should be [in this list](builder/osi-approved.yml) | `apache-2.0` |
| `avatar` | a picture that personalizes and distinguishes your image | None |
| `platform` | a list of CPU architectures that your image is built on, each item should be [in this list](builder/platforms.yml) | `[linux/amd64]` | 

Please refer to [hub/examples/mwu_encoder/manifest.yml](hub/examples/mwu_encoder/manifest.yml) for the example.

### Steps to Publish Your Image

All you need is to publish your bundle into this repo, the Docker image building, uploading and tagging are all handled automatically by our CICD pipeline. 

1. Let's say your organize [all files mentioned in here](#what-files-need-to-be-uploaded) in a folder called `awesomeness`. Depending on what you are contributing, you can put it into `hub/executors/indexers/awesomeness`.
2. Make a Pull Request and commit your changes into this repository, remember to follow the commit lint style.
3. Wait until the CICD finish and you get at least one reviewer's approval.
4. Merge it! 

The CICD pipeline will work on building, uploading and tagging the image on the Jina Hub.

The image will be available at `jinaai/hub.executors.indexers.awesomeness:0.0.0` assuming your version number is defined as `0.0.0` in `manifest.yml`.

You can use it as [we described here](#use-your-pod-image).  

### Why My Upload Fails on the CICD?

Click "Details" and checkout the log of the CICD pipeline:

![](.github/.README_images/5f4181e9.png)

Here is the checklist to help you locate the problem.

- [ ] The required field in `manifest.yml` is missing.
- [ ] Some field value is not in the correct format, not passing the sanity check.
- [ ] The pod bundle is badly placed.
- [ ] The build is success but it fails on [three basic usage tests](#use-your-pod-image).


## License

If you have downloaded a copy of the Jina binary or source code, please note that Jina's binary and source code are both licensed under the [Apache 2.0](LICENSE).







