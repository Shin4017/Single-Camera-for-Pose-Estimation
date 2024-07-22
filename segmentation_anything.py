import cv2
from PIL import Image
from ultralytics import YOLO
import numpy as np
import torch
import time

from skimage.draw import polygon2mask

model = YOLO('yolov8l-seg.pt').to(torch.device("cpu"))
cap = cv2.imread('2.jpg')
print(cap.shape)
# results = model.predict(source=cap, stream= True, )
results = model(source=cap)


# print(len(list(results)))
for result in results:
    # Detection
    # result.boxes.xyxy  # box with xyxy format, (N, 4)
    # result.boxes.xywh  # box with xywh format, (N, 4)
    # result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
    # result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
    # result.boxes.conf  # confidence score, (N, 1)
    # result.boxes.cls  # cls, (N, 1)

    # Segmentation
    # result.masks.data  # masks, (N, H, W)
    # result.masks.xy  # x,y segments (pixels), List[segment] * N)
    # result.masks.xyn  # x,y segments (normalized), List[segment] * N
    print("Results: {}".format(result.masks[0].shape))
    print("XY: {}".format(len(result.masks.xy)))
    # color = np.array([255, 255, 0], dtype=np.uint8)
    color_mask = np.zeros_like(cap, dtype=np.uint8)
    visualized_image = cap.copy()
    for i, ins in enumerate(result.masks.xy):
        # box = result.boxes.xyxy[i].numpy()
        # print("Class: {}".format(result.boxes.cls[i]))
        # print("Box: {}".format(box))
        color = np.array([255, 255, 0], dtype=np.uint8)
        print("Num points: {}".format(ins.shape))
        print("Max xy: {}".format(np.max(ins, axis=0)))
        # centroid = np.mean(ins, axis=0)
        polygon = ins[:, [1, 0]]
        mask = polygon2mask((cap.shape[0], cap.shape[1]), polygon)
        points = np.argwhere(mask == True)
        min_x = np.min(points[:, 1])
        max_x = np.max(points[:, 1])
        min_y = np.min(points[:, 0])
        max_y = np.max(points[:, 0])
        print("points: {}".format(points))
        centroid = np.mean(points, axis=0)[::-1]
        color_mask = np.where(mask[:, :, np.newaxis], color[np.newaxis, np.newaxis, :], color_mask)
        visualized_image = np.where(color_mask > 0, cv2.addWeighted(visualized_image, 2, color_mask, 0.4, 0), visualized_image)
        print("Mask: {}".format(mask.shape))
        print("Centroid xy: {}".format(centroid))
        cv2.circle(visualized_image, (int(centroid[0]), int(centroid[1])), radius=10, color=(0, 0, 255), thickness=-1)
        # x1, y1, x2, y2 = box
        # cv2.rectangle(visualized_image, (min_x, min_y), (max_x, max_y), (0, 0, 255), 3)
    
    cv2.imwrite("Anh.jpg", visualized_image)
    # cv2.waitKey(0)
    # for ins in result.masks.data:
    #     print(ins.numpy().shape)
    # for mask in result.masks:
    #     np_mask = mask.cpu()
    #     print(np.unique(np_mask.data.numpy()))
    #     # mask_coords = np.argwhere(np_mask.data.numpy().squeeze() == 1)
    #     # print("Mask coords: {}, xy: {}".format(mask_coords, len(np_mask.xy)))
    #     mask_coords = np_mask.xy[0][0]
    #     print("Mask coords: {}".format(mask.shape))


# Each result is composed of torch.Tensor by default,
# in which you can easily use following functionality:
# result = result.cuda()
# result = result.cpu()
# result = result.to("gpu")
# result = result.numpy()

# cv2.waitKey(0)
# cv2.destroyAllWindows()