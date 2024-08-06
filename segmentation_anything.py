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
        # color_mask = np.zeros_like(cap, dtype=np.uint8)
        # visualized_image = cap.copy()
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
        # color_mask = np.where(mask[:, :, np.newaxis], color[np.newaxis, np.newaxis, :], color_mask)
        # visualized_image = np.where(color_mask > 0, cv2.addWeighted(visualized_image, 2, color_mask, 0.4, 0), visualized_image)
        print("Mask: {}".format(mask.shape))
        print("Centroid xy: {}".format(centroid))
        # cv2.circle(visualized_image, (int(centroid[0]), int(centroid[1])), radius=10, color=(0, 0, 255), thickness=-1)
        # x1, y1, x2, y2 = box
        cv2.rectangle(visualized_image, (min_x, min_y), (max_x, max_y), (0, 0, 255), 3)

        dist_x = max_x - min_x
        dist_y = max_y - min_y

        if dist_x > dist_y:
            divide_by_x = True
        else:
            divide_by_x = False

        if divide_by_x:
            one_third_x = int(min_x + 1/3 * dist_x)
            two_third_x = int(min_x + 2/3 * dist_x)
            print("one_third_x: {}, two_third_x: {}, min_x: {}, max_x: {}".format(one_third_x, two_third_x, min_x, max_x))
            mask_one_third = mask.copy()
            mask_one_third[:, one_third_x:] = False
            mask_one_two = mask.copy()
            # mask_one_two[:, one_third_x:(two_third_x + 1)] = False
            mask_one_two[: :one_third_x] = False
            mask_one_two[:, two_third_x:] = False
            mask_two_third = mask.copy()
            mask_two_third[:, :two_third_x] = False
            area_one_third = np.argwhere(mask_one_third == True).shape[0]
            area_one_two = np.argwhere(mask_one_two == True).shape[0]
            area_two_third = np.argwhere(mask_two_third == True).shape[0]
            print(area_one_third, area_one_two, area_two_third)

            max_ind = np.argmax([area_one_third, area_one_two, area_two_third])
            if max_ind == 0:
                max_mask = mask_one_third
            elif max_ind == 1:
                max_mask = mask_one_two
            else:
                max_mask = mask_two_third
            print(max_ind)
            # chosen_point = np.mean(np.argwhere(max_mask == True), axis=0)[::-1]

        else:
            one_third_y = int(min_y + 1/3 * dist_y)
            two_third_y = int(min_y + 2/3 * dist_y)

            mask_one_third = mask.copy()
            mask_one_third[one_third_y:, :] = False
            mask_one_two = mask.copy()
            # mask_one_two[:, one_third_y:(two_third_y + 1)] = False
            mask_one_two[:one_third_y + 1, :] = False
            mask_one_two[two_third_y:, :] = False
            mask_two_third = mask.copy()
            mask_two_third[:two_third_y+1, :] = False
            area_one_third = np.argwhere(mask_one_third == True).shape[0]
            area_one_two = np.argwhere(mask_one_two == True).shape[0]
            area_two_third = np.argwhere(mask_two_third == True).shape[0]

            max_ind = np.argmax([area_one_third, area_one_two, area_two_third])
            if max_ind == 0:
                max_mask = mask_one_third
            elif max_ind == 1:
                max_mask = mask_one_two
            else:
                max_mask = mask_two_third

        chosen_point = np.mean(np.argwhere(max_mask == True), axis=0)[::-1]
        print("chosen_point: {}".format(chosen_point))
        
        # color = np.array([np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)], dtype=np.uint8)
        # color_mask = np.where(max_mask[:, :, np.newaxis], color[np.newaxis, np.newaxis, :], color_mask)
        if divide_by_x:
            color_mask = np.where(mask_one_third[:, :, np.newaxis], np.array([255, 0, 0], dtype=np.uint8)[np.newaxis, np.newaxis, :], color_mask)
            color_mask = np.where(mask_one_two[:, :, np.newaxis], np.array([0, 255, 0], dtype=np.uint8)[np.newaxis, np.newaxis, :], color_mask)
            color_mask = np.where(mask_two_third[:, :, np.newaxis], np.array([0, 0, 255], dtype=np.uint8)[np.newaxis, np.newaxis, :], color_mask)
            # visualized_image = np.where(color_mask > 0, cv2.addWeighted(visualized_image, 2, color_mask, 0.4, 0), visualized_image)

            cv2.circle(visualized_image, (int(chosen_point[0]), int(chosen_point[1])), radius=10, color=(0, 0, 255), thickness=-1)

        # cv2.imshow("Anh", visualized_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    visualized_image = np.where(color_mask > 0, cv2.addWeighted(visualized_image, 2, color_mask, 0.6, 0), visualized_image)
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