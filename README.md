# Jina Hub

Jina Hub is an open-registry for hosting Jina executors via container images. It enables users to ship and exchange reusable component across various Jina search applications.

From Jina 0.4.10, Jina Hub is referred as a Git Submodule in [`jina-ai/jina`](https://github.com/jina-ai/jina).

## Create a New Executor

```bash
pip install jina[hub]
jina hub new
```

It will start a wizard in CLI to guide you create your first executor. The resulted file structure should look like the following:

```text
- MyAwesomeExecutor/
    |
    |- Dockerfile
    |- manifest.yml
    |- README.md
    |- requirements.txt
    |- __init__.py
    |- tests/
        |- test_MyAwesomeExecutor.py
        |- __init__.py
```

## Test an Executor Locally

```bash
jina hub build /MyAwesomeExecutor/
```

More Hub CLI usage can be found via `jina hub build --help`

## Contributing

We welcome all kinds of contributions from the open-source community, individuals and partners. Without your active involvement, Jina won't be successful.

Please first read [the contributing guidelines](https://github.com/jina-ai/jina/blob/master/CONTRIBUTING.md) before the submission. 

## License

Copyright (c) 2020 Jina AI Limited. All rights reserved.

Jina is licensed under the Apache License, Version 2.0. [See LICENSE for the full license text.](LICENSE)
