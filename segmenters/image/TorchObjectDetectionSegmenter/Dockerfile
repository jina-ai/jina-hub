FROM jinaai/jina:1.3.0

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install the third-party requirements
RUN pip install -r requirements.txt

# for testing the image
RUN pip install pytest && pip install pytest-mock && pytest -v -s && rm -rf tests/imgs && \
rm -rf /root/.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth && \
rm -rf /root/.cache/torch/hub/checkpoints/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]