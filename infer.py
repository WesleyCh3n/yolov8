import trt
import cv2

model = trt.YOLOv8("./runs/detect/train2/weights/detect_13k.trt", 1)
result = model(["/media/cowdata/dataset/fonler-0203/1/20240220-093547.772_0_1.jpg"] * 3)
print(result)
img = cv2.imread("/media/cowdata/dataset/fonler-0203/1/20240220-093547.772_0_1.jpg")
for b in result[0]:
    cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), thickness=2)
cv2.imwrite("test.jpg", img)
