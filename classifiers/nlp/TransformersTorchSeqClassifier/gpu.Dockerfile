FROM continuumio/miniconda3 AS conda

COPY env_gpu.yml /
COPY requirements.txt /
RUN conda update conda -c conda-forge && \
    conda env update -f /env_gpu.yml -n base && \
    pip install -r  /requirements.txt --no-cache-dir && \
    conda clean -afy

FROM nvidia/cuda:11.0-base-ubuntu20.04 AS base

# Prepare shell and file system
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 SHELL=/bin/bash
ENV PATH /opt/conda/bin:$PATH
SHELL ["/bin/bash", "-c"]

# Commented out until GPU testing possible in CI
#ENV JINA_TEST_GPU=true

# Copy over conda and bashrc, install environment
COPY --from=conda /opt/ /opt/
RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# for testing the image
# FROM base

RUN pip install pytest && pytest

# FROM base

ENTRYPOINT ["jina", "pod", "--uses", "config.yml", "--timeout-ready", "180000"]
