FROM continuumio/miniconda3 as conda

COPY requirements.txt /
RUN conda update conda -c conda-forge && \
    pip install -r requirements.txt --no-cache-dir && \
    conda clean -afy

FROM nvidia/cuda:11.0-base-ubuntu20.04 as base

ENV LANG=C.UTF-8 LC_ALL=C-UTF-8 SHELL=/bin/bash
ENV PATH /opt/conda/bin:$PATH
SHELL ["/bin/bash", "-c"]

COPY --from=conda /opt/ /opt/
RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# for testing the image
FROM base as test
RUN pip install pytest && pytest

FROM base as final
ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]
