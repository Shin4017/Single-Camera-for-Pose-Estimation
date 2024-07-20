import cv2
from PIL import Image
from ultralytics import YOLO
import numpy as np
import torch

model = YOLO('yolov8l-seg.pt')

results = model.predict(source=0, stream=True)
for result in results:
    # Detection
    result.boxes.xyxy  # box with xyxy format, (N, 4)
    result.boxes.xywh  # box with xywh format, (N, 4)
    result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
    result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
    result.boxes.conf  # confidence score, (N, 1)
    result.boxes.cls  # cls, (N, 1)

    # Segmentation
    result.masks.data  # masks, (N, H, W)
    result.masks.xy  # x,y segments (pixels), List[segment] * N
    result.masks.xyn  # x,y segments (normalized), List[segment] * N

    sam_list = result.show()

    # Tìm trọng tâm hình
    def find_centroid(sam_list):
      # Tính tích phân tích lũy
      integral_image = cv2.integral(sam_list)

      # Tính tổng giá trị của tất cả các pixel
      total_value = integral_image[-1, -1]

      # Tính tổng moment x và y
      moment_x = integral_image[-1, :]
      moment_y = integral_image[:, -1]

      # Tính tọa độ trọng tâm
      centroid_x = np.sum(moment_x) / total_value
      centroid_y = np.sum(moment_y) / total_value

      return (int(centroid_x), int(centroid_y))

    # Tìm trọng tâm
    print(find_centroid)
    centroid = find_centroid(sam_list)
    
    cv2.imshow('Detect', centroid)

# Each result is composed of torch.Tensor by default,
# in which you can easily use following functionality:
result = result.cuda()
result = result.cpu()
result = result.to("gpu")
result = result.numpy()



# cv2.imshow('Image with Centroid', centroid)
# cv2.waitKey(0)
# cv2.destroyAllWindows()