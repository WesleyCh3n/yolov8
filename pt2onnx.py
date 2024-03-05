from ultralytics import YOLO

model = YOLO("./runs/detect/train2/weights/best.pt")
model.fuse()
model.info(verbose=False)  # Print model information
model.export(format="onnx", simplify=True, dynamic=True)
