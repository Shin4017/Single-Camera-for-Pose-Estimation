# import library
import cv2
from PIL import Image
from ultralytics import YOLO
import numpy as np
import torch
import time
from skimage.draw import polygon2mask

# input camera, model
# model = YOLO('yolov8l-seg.pt').to(torch.device("cpu"))
# cap = cv2.imread('test6.jpg')
# print(cap.shape)
# results = model(source=cap)


# Function segmentation, mask


# Function find center of object using hull_contour


# Function depth_estimation


# Show on screen

def calculate_distance_between_two_points(point1, point2) -> float:
    x1, y1 = point1
    x2, y2 = point2

    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

model = YOLO('yolov8l-seg.pt').to(torch.device("cpu"))
cap = cv2.imread('test1.jpg')
print(cap.shape)
# results = model.predict(source=cap, stream= True, )
results = model(source=cap)

for result in results:
    # print("Results: {}".format(result.masks[0].shape))
    # print("XY: {}".format(len(result.masks.xy)))
    # color = np.array([255, 255, 0], dtype=np.uint8)
    color_mask = np.zeros_like(cap, dtype=np.uint8)
    visualized_image = cap.copy()

    for i, ins in enumerate(result.masks.xy):
        # box = result.boxes.xyxy[i].numpy()
        # print("Class: {}".format(result.boxes.cls[i]))
        # print("Box: {}".format(box))

        # Find the center of the object
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

    contours_area = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        contours_area.append(area)

    max_contour = contours[np.argmax(contours_area)]

    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    print("box: {}".format(box))
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

        # # Find the object's grip points
        # west_index = np.argmin(polygon[:, 0])
        # west_point = polygon[west_index]
        # east_index = np.argmax(polygon[:, 0])
        # east_point = polygon[east_index]
        # south_index = np.argmin(polygon[:, 1])
        # south_point = polygon[south_index]
        # north_index = np.argmax(polygon[:, 1])
        # north_point = polygon[north_index]
        # print("west_point: {}, east_point: {}, south_point: {}, south_point: {}".format(west_point, east_point, south_point, north_point))
        # cv2.line(visualized_image, (int(west_point[0]), 0), (int(west_point[0]), visualized_image.shape[0]), (0, 0, 255), 2)
        # cv2.line(visualized_image, (int(east_point[0]), 0), (int(east_point[0]), visualized_image.shape[0]), (0, 0, 255), 2) 
        # cv2.line(visualized_image, (0, int(south_point[1])), (visualized_image.shape[1], int(south_point[1])), (0, 0, 255), 2)
        # cv2.line(visualized_image, (0, int(north_point[1])), (visualized_image.shape[1], int(north_point[1])), (0, 0, 255), 2)

    cv2.imwrite("Anh.jpg", visualized_image)
    cv2.imwrite("Mask.jpg", image_mask)
        # # x1, y1, x2, y2 = box
        # cv2.rectangle(visualized_image, (min_x, min_y), (max_x, max_y), (0, 0, 255), 3)

        # dist_x = max_x - min_x
        # dist_y = max_y - min_y

        # if dist_x > dist_y:
        #     divide_by_x = True
        # else:
        #     divide_by_x = False

        # if divide_by_x:
        #     one_third_x = int(min_x + 1/3 * dist_x)
        #     two_third_x = int(min_x + 2/3 * dist_x)
        #     print("one_third_x: {}, two_third_x: {}, min_x: {}, max_x: {}".format(one_third_x, two_third_x, min_x, max_x))
        #     mask_one_third = mask.copy()
        #     mask_one_third[:, one_third_x:] = False
        #     mask_one_two = mask.copy()
        #     # mask_one_two[:, one_third_x:(two_third_x + 1)] = False
        #     mask_one_two[: :one_third_x] = False
        #     mask_one_two[:, two_third_x:] = False
        #     mask_two_third = mask.copy()
        #     mask_two_third[:, :two_third_x] = False
        #     area_one_third = np.argwhere(mask_one_third == True).shape[0]
        #     area_one_two = np.argwhere(mask_one_two == True).shape[0]
        #     area_two_third = np.argwhere(mask_two_third == True).shape[0]
        #     print(area_one_third, area_one_two, area_two_third)

        #     max_ind = np.argmax([area_one_third, area_one_two, area_two_third])
        #     if max_ind == 0:
        #         max_mask = mask_one_third
        #     elif max_ind == 1:
        #         max_mask = mask_one_two
        #     else:
        #         max_mask = mask_two_third
        #     print(max_ind)
        #     # chosen_point = np.mean(np.argwhere(max_mask == True), axis=0)[::-1]

        # else:
        #     one_third_y = int(min_y + 1/3 * dist_y)
        #     two_third_y = int(min_y + 2/3 * dist_y)

        #     mask_one_third = mask.copy()
        #     mask_one_third[one_third_y:, :] = False
        #     mask_one_two = mask.copy()
        #     # mask_one_two[:, one_third_y:(two_third_y + 1)] = False
        #     mask_one_two[:one_third_y + 1, :] = False
        #     mask_one_two[two_third_y:, :] = False
        #     mask_two_third = mask.copy()
        #     mask_two_third[:two_third_y+1, :] = False
        #     area_one_third = np.argwhere(mask_one_third == True).shape[0]
        #     area_one_two = np.argwhere(mask_one_two == True).shape[0]
        #     area_two_third = np.argwhere(mask_two_third == True).shape[0]

        #     max_ind = np.argmax([area_one_third, area_one_two, area_two_third])
        #     if max_ind == 0:
        #         max_mask = mask_one_third
        #     elif max_ind == 1:
        #         max_mask = mask_one_two
        #     else:
        #         max_mask = mask_two_third

        # chosen_point = np.mean(np.argwhere(max_mask == True), axis=0)[::-1]
        # print("chosen_point: {}".format(chosen_point))
        
        # # color = np.array([np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)], dtype=np.uint8)
        # # color_mask = np.where(max_mask[:, :, np.newaxis], color[np.newaxis, np.newaxis, :], color_mask)
        # if divide_by_x:
        #     color_mask = np.where(mask_one_third[:, :, np.newaxis], np.array([255, 0, 0], dtype=np.uint8)[np.newaxis, np.newaxis, :], color_mask)
        #     color_mask = np.where(mask_one_two[:, :, np.newaxis], np.array([0, 255, 0], dtype=np.uint8)[np.newaxis, np.newaxis, :], color_mask)
        #     color_mask = np.where(mask_two_third[:, :, np.newaxis], np.array([0, 0, 255], dtype=np.uint8)[np.newaxis, np.newaxis, :], color_mask)
        #     # visualized_image = np.where(color_mask > 0, cv2.addWeighted(visualized_image, 2, color_mask, 0.4, 0), visualized_image)

        #     cv2.circle(visualized_image, (int(chosen_point[0]), int(chosen_point[1])), radius=10, color=(0, 0, 255), thickness=-1)

        # cv2.imshow("Anh", visualized_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
    # visualized_image = np.where(color_mask > 0, cv2.addWeighted(visualized_image, 2, color_mask, 0.6, 0), visualized_image)
    # cv2.imwrite("Anh.jpg", visualized_image)
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