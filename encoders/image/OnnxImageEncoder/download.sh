#!/usr/bin/env bash

MODEL_NAME="mobilenetv2-1.0"
MODEL_DIR="pretrained"
mkdir -p ${MODEL_DIR}

curl -v https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/ --output ${MODEL_DIR}/mobilenetv2-1.0