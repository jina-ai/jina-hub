<p align="center">
<img src="https://github.com/jina-ai/jina-hub/blob/master/.github/Hub.png?raw=true" alt="Jina banner" width="200px">
</p>
<p align="center">
Jina's one-stop registry for hosting Pods and enabling easy neural search on the cloud
</p>
<p align=center>
</p>


<p align="center">
<a href="http://docs.jina.ai">Docs</a> • <a href="https://github.com/jina-ai/examples">Examples</a> • <a href="#contributing">Contribute</a> • <a href="https://jobs.jina.ai">Jobs</a> • <a href="http://jina.ai">Website</a> • <a href="http://slack.jina.ai">Slack</a>
</p>

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->



<!-- END doctoc generated TOC please keep comment here to allow auto update -->


## Use Hub Image in Flow API

```python
# my_image_tag is the built image name

from jina import Flow
with Flow().add(uses='docker://' + my_image_tag):
    pass
```

## Create a new Hub Pod/App

### Prerequisites

- Install [Docker](https://docs.docker.com/get-docker/).
- `pip install "jina[hub]"`

### 📦 Create a new Executor

```bash
jina hub new --type pod
```

It will start a wizard in CLI to guide you create your first executor. The resulted file structure should look like the following:

```text
MyAwesomeExecutor/
├── Dockerfile
├── manifest.yml
├── config.yml
├── README.md
├── requirements.txt
├── __init__.py
└── tests/
    ├── test_MyAwesomeExecutor.py
    └── __init__.py

```
Link to the documentation for creating a containerized pod is [here](https://docs.jina.ai/chapters/hub/build-pod-image.html).

### Create a new App

```bash
jina hub new --type app
```

It will start a wizard in CLI to guide you create your app.

## Upload your Image to Jina Hub

Build your Docker image using our [naming conventions](#naming-conventions):

```bash
jina hub build -t jinahub/type.kind.jina-image-name:image_version-jina_version <your_folder>
```

Push to Jina Hub:

`jina hub login`, then copy/paste the token into GitHub to verify your account

```bash
jina hub push jinahub/type.kind.jina-image-name:image-jina_version 
```

### Use `py_modules` to import multiple files

By default, `jina hub new --pod` creates a Python module structure and guides you to write `MyAwesomeExecutor` class into `__init__.py`. If you have some other files that need to be imported for `MyAwesomeExecutor`, say `helper.py`, you can change [`metas.pymodules`](https://docs.jina.ai/api/jina.executors.metas.html?highlight=py_modules#confval-py_modules) in `config.yml` to import those files. Note, you have to write the dependencies in reverse order. That is, if `__init__.py` depends on `A.py`, which again depends on `B.py`, then you need to write:

```yaml
!MyAwesomeExecutor
with:
  ...
metas:
  py_modules:
    - B.py
    - A.py
    - __init__.py
```

#### Legacy Hub Pod structure

This is also a valid structure, it works but not recommended:


```text
- MyAwesomeExecutor/
    |
    |- Dockerfile
    |- manifest.yml
    |- README.md
    |- requirements.txt
    |- MyAwesomeExecutor.py
    |- helper.py
    |- config.yml
        |- metas.py_modules
            |- helper.py
            |- MyAwesomeExecutor.py
```

Please note that:
    
- Here `MyAwesomeExecutor` directory is not a python module, as it lacks `__init__.py` under the root;
- to import ``foo.py``, you must to use ``from jinahub.foo import bar``, where ``jinahub`` is the common namespace for all external modules;
- in `config.yml:metas.py_modules`, ``helper.py`` needs to be put before `MyAwesomeExecutor.py`.

### Test a Pod/App locally

```bash
jina hub build /MyAwesomeExecutor/
```

More Hub CLI usage can be found via `jina hub build --help`

## Work with your own repository

1. Create a new repository
2. `pip install "jina[hub]" && jina hub new --type pod`
3. 
```
git checkout -b feat-new-executor
git add .
git commit -m "feat: add new executor"
git push
```
4. Add the [Hub Updater](https://github.com/jina-ai/action-hub-updater) and [Hub Builder](https://github.com/jina-ai/action-hub-builder) Github Actions to the Github workflow of your hub-type repo.
5. Make a Pull Request on `feat-new-executor -> master`


## Reference

### `manifest.yml` Schema

`manifest.yml` must exist if you want to publish your Pod image to Jina Hub.

`manifest.yml` annotates your image so that it can be managed by Jina Hub. To get better appealing on Jina Hub, you should carefully set `manifest.yml` to the correct values.

| Key | Description | Default |
| --- | --- | --- |
| `manifest_version` | The version of the manifest protocol | `1` |
| `type` | The type of the image | Required |
| `kind` | The kind of the executor | Required |
| `name` | Human-readable title of the image (must match with `^[a-zA-Z_$][a-zA-Z_\s\-$0-9]{3,30}$`) | Required |
| `description` | Human-readable description of the software packaged in the image | Required |
| `author` | Contact details of the people or organization responsible for the image (string) | `Jina AI Dev-Team (dev-team@jina.ai)` |
| `url` | URL to find more information on the image (string) | `https://jina.ai` |
| `documentation` | URL to get documentation on the image (string) | `https://docs.jina.ai` |
| `version` | Version of the image, it should be [Semantic versioning-compatible](http://semver.org/) | `0.0.0` |
| `vendor` | The name of the distributing entity, organization or individual (string) | `Jina AI Limited` |
| `license` | License under which contained software is distributed, it should be [in this list](legacy/builder/osi-approved.yml) | `apache-2.0` |
| `avatar` | A picture that personalizes and distinguishes your image | None |
| `platform` | A list of CPU architectures that your image built on, each item should be [in this list](legacy/builder/platforms.yml) | `[linux/amd64]` |
| `keywords` | A list of strings help user to filter and locate your package  | None | 

### Naming conventions

All apps and executors should follow the naming convention:

```
jinahub/type.kind.jina_image_name:image_version-jina_version
```

For example:

```
jinahub/pod.encoder.transformertfencoder:0.0.16-0.9.33
```

| Text                           | Meaning                       |
| ---                            | ---                           |
| `jinahub/`                     | Push to `jinahub` Docker repo |
| `app`                          | Type                      |
| `encoder`                          | Kind                      |
| `transformertfencoder`         | Pod name                      |
| `0.0.16`                       | Pod version                   |
| `0.9.33`                       | Jina version                  |

From Jina 0.4.10 onwards, Jina Hub is referred as a Git Submodule in [`jina-ai/jina`](https://github.com/jina-ai/jina).

## Contributing

We welcome all kinds of contributions from the open-source community, individuals and partners. Without your active involvement, Jina won't be successful.

Please first read [the contributing guidelines](https://github.com/jina-ai/jina/blob/master/CONTRIBUTING.md) before the submission.

### Contributors ✨

<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-113-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->


<kbd><a href="https://jina.ai/"><img src="https://avatars1.githubusercontent.com/u/61045304?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="http://weizhen.rocks/"><img src="https://avatars3.githubusercontent.com/u/5943684?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/phamtrancsek12"><img src="https://avatars3.githubusercontent.com/u/14146667?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/gsajko"><img src="https://avatars1.githubusercontent.com/u/42315895?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://t.me/neural_network_engineering"><img src="https://avatars1.githubusercontent.com/u/1935623?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://hanxiao.io/"><img src="https://avatars2.githubusercontent.com/u/2041322?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/YueLiu-jina"><img src="https://avatars1.githubusercontent.com/u/64522311?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/nan-wang"><img src="https://avatars3.githubusercontent.com/u/4329072?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/tracy-propertyguru"><img src="https://avatars2.githubusercontent.com/u/47736458?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://www.linkedin.com/in/maanavshah/"><img src="https://avatars0.githubusercontent.com/u/30289560?v=4" class="avatar-user" width="32px;"/></a></kbd>
<kbd><a href="https://github.com/iego2017"><img src="https://avatars3.githubusercontent.com/u/44792649?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://www.davidsanwald.net/"><img src="https://avatars1.githubusercontent.com/u/10153003?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="http://alexcg1.github.io/"><img src="https://avatars2.githubusercontent.com/u/4182659?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/shivam-raj"><img src="https://avatars3.githubusercontent.com/u/43991882?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="http://dncc.github.io/"><img src="https://avatars1.githubusercontent.com/u/126445?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="http://johnarevalo.github.io/"><img src="https://avatars3.githubusercontent.com/u/1301626?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/imsergiy"><img src="https://avatars3.githubusercontent.com/u/8855485?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://guiferviz.com/"><img src="https://avatars2.githubusercontent.com/u/11474949?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/rohan1chaudhari"><img src="https://avatars1.githubusercontent.com/u/9986322?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://www.linkedin.com/in/mohong-pan/"><img src="https://avatars0.githubusercontent.com/u/45755474?v=4" class="avatar-user" width="32px;"/></a></kbd>
<kbd><a href="https://github.com/anish2197"><img src="https://avatars2.githubusercontent.com/u/16228282?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/joanna350"><img src="https://avatars0.githubusercontent.com/u/19216902?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://www.linkedin.com/in/madhukar01"><img src="https://avatars0.githubusercontent.com/u/15910378?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/maximilianwerk"><img src="https://avatars0.githubusercontent.com/u/4920275?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/emmaadesile"><img src="https://avatars2.githubusercontent.com/u/26192691?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/YikSanChan"><img src="https://avatars1.githubusercontent.com/u/17229109?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/Zenahr"><img src="https://avatars1.githubusercontent.com/u/47085752?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/JoanFM"><img src="https://avatars3.githubusercontent.com/u/19825685?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="http://yangboz.github.io/"><img src="https://avatars3.githubusercontent.com/u/481954?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/boussoffara"><img src="https://avatars0.githubusercontent.com/u/10478725?v=4" class="avatar-user" width="32px;"/></a></kbd>
<kbd><a href="https://github.com/fhaase2"><img src="https://avatars2.githubusercontent.com/u/44052928?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/Morriaty-The-Murderer"><img src="https://avatars3.githubusercontent.com/u/12904434?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/rutujasurve94"><img src="https://avatars1.githubusercontent.com/u/9448002?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/theUnkownName"><img src="https://avatars0.githubusercontent.com/u/3002344?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/vltmn"><img src="https://avatars3.githubusercontent.com/u/8930322?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/Kavan72"><img src="https://avatars3.githubusercontent.com/u/19048640?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/bwanglzu"><img src="https://avatars1.githubusercontent.com/u/9794489?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/antonkurenkov"><img src="https://avatars2.githubusercontent.com/u/52166018?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/redram"><img src="https://avatars3.githubusercontent.com/u/1285370?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/ericsyh"><img src="https://avatars3.githubusercontent.com/u/10498732?v=4" class="avatar-user" width="32px;"/></a></kbd>
<kbd><a href="https://github.com/festeh"><img src="https://avatars1.githubusercontent.com/u/6877858?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="http://julielab.de/Staff/Erik+F%C3%A4%C3%9Fler.html"><img src="https://avatars1.githubusercontent.com/u/4648560?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://www.cnblogs.com/callyblog/"><img src="https://avatars2.githubusercontent.com/u/30991932?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/JamesTang-jinaai"><img src="https://avatars3.githubusercontent.com/u/69177855?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/coolmian"><img src="https://avatars3.githubusercontent.com/u/36444522?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="http://www.joaopalotti.com/"><img src="https://avatars2.githubusercontent.com/u/852343?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://www.linkedin.com/in/deepankar-mahapatro/"><img src="https://avatars.githubusercontent.com/u/9050737?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/seraco"><img src="https://avatars.githubusercontent.com/u/25517036?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/SirsikarAkshay"><img src="https://avatars.githubusercontent.com/u/19791969?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/doomdabo"><img src="https://avatars.githubusercontent.com/u/72394295?v=4" class="avatar-user" width="32px;"/></a></kbd>
<kbd><a href="https://github.com/alasdairtran"><img src="https://avatars.githubusercontent.com/u/10582768?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/davidbp"><img src="https://avatars.githubusercontent.com/u/4223580?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/fernandakawasaki"><img src="https://avatars.githubusercontent.com/u/50497814?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/tadej-redstone"><img src="https://avatars.githubusercontent.com/u/69796623?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://www.linkedin.com/in/carlosbaezruiz/"><img src="https://avatars.githubusercontent.com/u/1107703?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/bhavsarpratik"><img src="https://avatars.githubusercontent.com/u/23080576?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/umbertogriffo"><img src="https://avatars.githubusercontent.com/u/1609440?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/anshulwadhawan"><img src="https://avatars.githubusercontent.com/u/25061477?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/fayeah"><img src="https://avatars.githubusercontent.com/u/29644978?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://sebastianlettner.info/"><img src="https://avatars.githubusercontent.com/u/51201318?v=4" class="avatar-user" width="32px;"/></a></kbd>
<kbd><a href="https://github.com/florian-hoenicke"><img src="https://avatars.githubusercontent.com/u/11627845?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/NouiliKh"><img src="https://avatars.githubusercontent.com/u/22430520?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://www.cnblogs.com/callyblog/"><img src="https://avatars.githubusercontent.com/u/30991932?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/pswu11"><img src="https://avatars.githubusercontent.com/u/48913707?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/BastinJafari"><img src="https://avatars.githubusercontent.com/u/25417797?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="http://bit.ly/3qKM0uO"><img src="https://avatars.githubusercontent.com/u/13751208?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/kaushikb11"><img src="https://avatars.githubusercontent.com/u/45285388?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/ApurvaMisra"><img src="https://avatars.githubusercontent.com/u/22544948?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/serge-m"><img src="https://avatars.githubusercontent.com/u/4344566?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://educatorsrlearners.github.io/portfolio.github.io/"><img src="https://avatars.githubusercontent.com/u/17770276?v=4" class="avatar-user" width="32px;"/></a></kbd>
<kbd><a href="https://github.com/Arrrlex"><img src="https://avatars.githubusercontent.com/u/13290269?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/janandreschweiger"><img src="https://avatars.githubusercontent.com/u/44372046?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://kilsenp.github.io/"><img src="https://avatars.githubusercontent.com/u/5173119?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://www.linkedin.com/in/prabhupad-pradhan/"><img src="https://avatars.githubusercontent.com/u/11462012?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/averkij"><img src="https://avatars.githubusercontent.com/u/1473991?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/Roshanjossey"><img src="https://avatars.githubusercontent.com/u/8488446?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/bio-howard"><img src="https://avatars.githubusercontent.com/u/74507907?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/HelioStrike"><img src="https://avatars.githubusercontent.com/u/34064492?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/rjgallego"><img src="https://avatars.githubusercontent.com/u/59635994?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/ThePfarrer"><img src="https://avatars.githubusercontent.com/u/7157861?v=4" class="avatar-user" width="32px;"/></a></kbd>
<kbd><a href="https://github.com/pgiank28"><img src="https://avatars.githubusercontent.com/u/17511966?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/clennan"><img src="https://avatars.githubusercontent.com/u/19587525?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/xinbinhuang"><img src="https://avatars.githubusercontent.com/u/27927454?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/jancijen"><img src="https://avatars.githubusercontent.com/u/28826229?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/Showtim3"><img src="https://avatars.githubusercontent.com/u/30312043?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/JamesTang-616"><img src="https://avatars.githubusercontent.com/u/69177855?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/Immich"><img src="https://avatars.githubusercontent.com/u/9353470?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/smy0428"><img src="https://avatars.githubusercontent.com/u/61920576?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/fsal"><img src="https://avatars.githubusercontent.com/u/9203508?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://sreerag-ibtl.github.io/"><img src="https://avatars.githubusercontent.com/u/39914922?v=4" class="avatar-user" width="32px;"/></a></kbd>
<kbd><a href="https://github.com/tadejsv"><img src="https://avatars.githubusercontent.com/u/11489772?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://blog.lsgrep.com/"><img src="https://avatars.githubusercontent.com/u/3893940?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/deepampatel"><img src="https://avatars.githubusercontent.com/u/19245659?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://www.linkedin.com/in/nicholas-cwh/"><img src="https://avatars.githubusercontent.com/u/25291155?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/YueLiu1415926"><img src="https://avatars.githubusercontent.com/u/64522311?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://www.linkedin.com/in/yuanb/"><img src="https://avatars.githubusercontent.com/u/12972261?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/hongchhe"><img src="https://avatars.githubusercontent.com/u/25891193?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://cristianmtr.github.io/resume/"><img src="https://avatars.githubusercontent.com/u/8330330?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/jyothishkjames"><img src="https://avatars.githubusercontent.com/u/937528?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://www.linkedin.com/in/lucia-loher/"><img src="https://avatars.githubusercontent.com/u/64148900?v=4" class="avatar-user" width="32px;"/></a></kbd>
<kbd><a href="https://github.com/RenrakuRunrat"><img src="https://avatars.githubusercontent.com/u/14925249?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/dalekatwork"><img src="https://avatars.githubusercontent.com/u/40423996?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/ManudattaG"><img src="https://avatars.githubusercontent.com/u/8463344?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/harry-stark"><img src="https://avatars.githubusercontent.com/u/43717480?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/Yongxuanzhang"><img src="https://avatars.githubusercontent.com/u/44033547?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://www.linkedin.com/in/amrit3701/"><img src="https://avatars.githubusercontent.com/u/10414959?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/yk"><img src="https://avatars.githubusercontent.com/u/858040?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/FionnD"><img src="https://avatars.githubusercontent.com/u/59612379?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://www.imxiqi.com/"><img src="https://avatars.githubusercontent.com/u/4802250?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/samjoy"><img src="https://avatars.githubusercontent.com/u/3750744?v=4" class="avatar-user" width="32px;"/></a></kbd>
<kbd><a href="https://prasakis.com/"><img src="https://avatars.githubusercontent.com/u/10392953?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/PabloRN"><img src="https://avatars.githubusercontent.com/u/727564?v=4" class="avatar-user" width="32px;"/></a></kbd> <kbd><a href="https://github.com/rameshwara"><img src="https://avatars.githubusercontent.com/u/13378629?v=4" class="avatar-user" width="32px;"/></a></kbd>


<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

## Community

- [Code of conduct](https://github.com/jina-ai/jina/blob/master/.github/CODE_OF_CONDUCT.md) - play nicely with the Jina community
- [Slack workspace](https://slack.jina.ai) - join #general on our Slack to meet the team and ask questions
- [YouTube channel](https://youtube.com/c/jina-ai) - subscribe to the latest video tutorials, release demos, webinars and presentations.
- [LinkedIn](https://www.linkedin.com/company/jinaai/) - get to know Jina AI as a company and find job opportunities
- [![Twitter Follow](https://img.shields.io/twitter/follow/JinaAI_?label=Follow%20%40JinaAI_&style=social)](https://twitter.com/JinaAI_) - follow and interact with us using hashtag `#JinaSearch`
- [Company](https://jina.ai) - know more about our company and how we are fully committed to open-source.

## Open Governance

As part of our open governance model, we host Jina's [Engineering All Hands]((https://hanxiao.io/2020/08/06/Engineering-All-Hands-in-Public/)) in public. This Zoom meeting recurs monthly on the second Tuesday of each month, at 14:00-15:30 (CET). Everyone can join in via the following calendar invite.

- [Add to Google Calendar](https://calendar.google.com/event?action=TEMPLATE&tmeid=MHIybG03cjAwaXE3ZzRrYmVpaDJyZ2FpZjlfMjAyMDEwMTNUMTIwMDAwWiBjXzF0NW9nZnAyZDQ1djhmaXQ5ODFqMDhtY200QGc&tmsrc=c_1t5ogfp2d45v8fit981j08mcm4%40group.calendar.google.com&scp=ALL)
- [Download .ics](https://hanxiao.io/2020/08/06/Engineering-All-Hands-in-Public/jina-ai-public.ics)

The meeting will also be live-streamed and later published to our [YouTube channel](https://youtube.com/c/jina-ai).

## Join us

Jina is an open-source project. [We are hiring](https://jobs.jina.ai) full-stack developers, evangelists, and PMs to build the next neural search ecosystem in open source.

## License

Copyright (c) 2020 Jina AI Limited. All rights reserved.

Jina is licensed under the Apache License, Version 2.0. [See LICENSE for the full license text.](LICENSE)
