## YOLO v8

```sh
yolo detect train data=./yolo-yaml/cowface-15k.yaml model=yolov8n.pt epochs=100

python yolo_pt_to_onnx.py

sudo docker run --rm -it --gpus all \
    -v $PWD:/workspace \
    nvcr.io/nvidia/tensorrt:23.08-py3 /bin/bash

# in container
# for normal trt
trtexec  --onnx=best.onnx \
    --explicitBatch \
    --minShapes=images:1x3x640x640 \
    --optShapes=images:1x3x640x640 \
    --maxShapes=images:64x3x640x640 \
    --saveEngine=best.trt
# for fp16 trt
trtexec --explicitBatch --onnx=best.onnx\
    --minShapes=images:1x3x640x640\
    --maxShapes=images:64x3x640x640\
    --fp16 \
    --saveEngine=best-fp16.trt
```

