FROM jinaai/jina:devel
ADD shufflenet_v2.yml ./
RUN apt-get update && apt-get install libgtk2.0-dev -y
RUN pip3 install opencv-python common dual tight data prox \
    protobuf==3.11.0 six==1.14.0 matplotlib==3.1.3
RUN pip3 install paddle paddlepaddle paddlehub
COPY . /
ENTRYPOINT ["jina", "pod", "--yaml-path", "shufflenet_v2.yml", "--timeout-ctrl", "-1", "--timeout-ready", "-1"]
