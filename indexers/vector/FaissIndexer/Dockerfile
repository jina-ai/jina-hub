FROM jinaai/jina

COPY . /workspace
WORKDIR /workspace

# adapting from miniconda image
RUN apt-get update && apt-get install --no-install-recommends -y curl bzip2 && \
    curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -bfp /usr/local/ && \
    rm -rf /tmp/miniconda.sh && \
    conda install -y python=3 && \
    conda update conda && \
    conda clean --all --yes && \
    apt-get remove -y curl bzip2 && \
    apt-get clean

RUN conda update -n base conda && \
    conda install -y -c pytorch faiss-cpu && \
    conda install -c anaconda setuptools && \
    conda install pip

RUN pip install -r requirements.txt
RUN pip install pytest && pytest tests/ -v -s

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]