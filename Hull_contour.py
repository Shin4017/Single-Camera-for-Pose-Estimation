# import library
import cv2
from PIL import Image
from ultralytics import YOLO
import numpy as np
import torch
import time
from skimage.draw import polygon2mask

# input camera, model
def calculate_distance_between_two_points(point1, point2) -> float:
    x1, y1 = point1
    x2, y2 = point2

    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

model = YOLO('yolov8l-seg.pt').to(torch.device("cpu"))
cap = cv2.imread('data3.jpg')
print(cap.shape)
# results = model.predict(source=cap, stream= True, )
results = model(source=cap)

# Tìm kiếm trung điểm của vật
for result in results:
    color_mask = np.zeros_like(cap, dtype=np.uint8)
    visualized_image = cap.copy()

    for i, ins in enumerate(result.masks.xy):
        color = np.array([255, 255, 0], dtype=np.uint8)
        print("Num points: {}".format(ins.shape))
        print("Max xy: {}".format(np.max(ins, axis=0)))
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

    image_mask = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(image_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # contours_area = []
    # for cnt in contours:
    #     area = cv2.contourArea(cnt)
    #     contours_area.append(area)

    # max_contour = contours[np.argmax(contours_area)]

    # rect = cv2.minAreaRect(max_contour)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # print("box: {}".format(box))
    # cv2.drawContours(visualized_image, [box], -1, (0, 255, 0), 5)

    # x1, y1 = box[0]
    # x2, y2 = box[1]
    # x3, y3 = box[2]
    # x4, y4 = box[3]

    # print(calculate_distance_between_two_points((x1, y1), (x2, y2)))
    # print(calculate_distance_between_two_points((x3, y3), (x4, y4)))
    # print(calculate_distance_between_two_points((x2, y2), (x3, y3)))
    # print(calculate_distance_between_two_points((x1, y1), (x4, y4)))

    # assert np.allclose(calculate_distance_between_two_points((x1, y1), (x2, y2)), calculate_distance_between_two_points((x3, y3), (x4, y4)))
    # assert np.allclose(calculate_distance_between_two_points((x2, y2), (x3, y3)), calculate_distance_between_two_points((x1, y1), (x4, y4)))

    # width = calculate_distance_between_two_points((x1, y1), (x2, y2))
    # height = calculate_distance_between_two_points((x1, y1), (x4, y4))

    # if width > height:
    #     point1 = ((3 * x1 + x2) / 4, (3 * y1 + y2) / 4)
    #     point2 = ((x1 + 3 * x2) / 4, (y1 + 3 * y2) / 4)
    #     point3 = ((3 * x3 + x4) / 4, (3 * y3 + y4) / 4)
    #     point4 = ((x3 + 3 * x4) / 4, (y3 + 3 * y4) / 4)

    # else:
    #     point1 = ((3 * x2 + x3) / 4, (3 * y2 + y3) / 4)
    #     point2 = ((x2 + 3 * x3) / 4, (y2 + 3 * y3) / 4)
    #     point3 = ((3 * x1 + x4) / 4, (3 * y1 + y4) / 4)
    #     point4 = ((x1 + 3 * x4) / 4, (y1 + 3 * y4) / 4)

    
    # 4 điểm chấm trên bounding box
    # cv2.circle(visualized_image, (int(point1[0]), int(point1[1])), radius=10, color=(0, 0, 255), thickness=-1)
    # cv2.circle(visualized_image, (int(point2[0]), int(point2[1])), radius=10, color=(0, 0, 255), thickness=-1)
    # cv2.circle(visualized_image, (int(point3[0]), int(point3[1])), radius=10, color=(0, 0, 255), thickness=-1)
    # cv2.circle(visualized_image, (int(point4[0]), int(point4[1])), radius=10, color=(0, 0, 255), thickness=-1)

    # 2 đường chéo trên vật
    # cv2.line(visualized_image, (int(point1[0]), int(point1[1])), (int(point3[0]), int(point3[1])), (0, 0, 255), 3)
    # cv2.line(visualized_image, (int(point2[0]), int(point2[1])), (int(point4[0]), int(point4[1])), (0, 0, 255), 3)

    # cv2.imwrite("Anh.jpg", visualized_image)
    # cv2.imwrite("Mask.jpg", image_mask)

cv2.imwrite("Anh.jpg", visualized_image)
cv2.imwrite("Mask.jpg", image_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()