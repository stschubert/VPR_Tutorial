import numpy as np

def iou(bbox_1: list[int, int, int, int], bbox_2: list[int, int, int, int]):
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(bbox_1.T)
    area2 = box_area(bbox_2.T)

    bbox_1 = bbox_1.squeeze()
    bbox_2 = bbox_2.squeeze()

    inter_width = np.min((bbox_1[3], bbox_2[3])) - np.max((bbox_1[1], bbox_2[1]))
    inter_height = np.min((bbox_1[2], bbox_2[2])) - np.max((bbox_1[0], bbox_2[0]))

    if inter_width <=0 or inter_height <= 0:
        return 0
    
    inter_area = inter_width * inter_height
    union_area = (area1 + area2) - inter_area
    # inter = (np.min(bbox_1[:, None, 2:], bbox_1[:, 2:]) -
    #          np.max(bbox_2[:, None, :2], bbox_2[:, :2])).clamp(0).prod(2)
    # iou = inter / (area1 + area2 - inter)
    return inter_area / union_area

def xywh2xyxy(x: list[int, int, int, int]) -> list[int, int, int, int]:
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return np.round(y).astype(int)

def xywh2xyxy_test(x: list[int, int, int, int]) -> list[int, int, int, int]:
    y = np.copy(x)
    y[:, 0] = x[:, 0]  # top left x
    y[:, 1] = x[:, 1]  # top left y
    y[:, 2] = x[:, 0] + x[:, 2]  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3]  # bottom right y

    #Why do I need to check if the order is correct?
    if y[:,2] < y[:,0]:
        new_values = (np.copy(y[:,0]), np.copy(y[:,2]))
        y[:,0] = new_values[1]
        y[:,2] = new_values[0]
    if y[:,3] < y[:,1]:
        new_values = (np.copy(y[:,1]), np.copy(y[:,3]))
        y[:,1] = new_values[1]
        y[:,3] = new_values[0]

    return np.round(y).astype(int)

def xyxy2xywh(x):
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return np.round(y).astype(int)

def normalise_xyxy(x, w, h):
    y = np.copy(x)
    y[:, 0] = int(y[:, 0] / w)
    y[:, 1] = int(y[:, 1] / h)
    y[:, 2] = int(y[:, 2] / w)
    y[:, 3] = int(y[:, 3] / h)
    return np.round(y).astype(int)

def xywhn2xyxy(x, w=640, h=640):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x).astype(int)
    y[:, 0] = _limit_num(int(np.round(w * (x[:, 0] - x[:, 2] / 2))), 0, w)  # top left x
    y[:, 1] = _limit_num(int(np.round(h * (x[:, 1] - x[:, 3] / 2))), 0, h) # top left y
    y[:, 2] = _limit_num(int(np.round(w * (x[:, 0] + x[:, 2] / 2))), 0, w)  # bottom right x
    y[:, 3] = _limit_num(int(np.round(h * (x[:, 1] + x[:, 3] / 2))), 0, h)  # bottom right y
    return np.round(y).astype(int)

def resize_bbox(bbox_1, bbox_2):
    bbox_1_height = (bbox_1[:,2] - bbox_1[:,0])
    bbox_1_width = (bbox_1[:,3] - bbox_1[:,1])
    bbox_1_area = bbox_1_height * bbox_1_width

    bbox_2_height = (bbox_2[:,2] - bbox_2[:,0])
    bbox_2_width = (bbox_2[:,3] - bbox_2[:,1])
    bbox_2_area = bbox_2_height * bbox_2_width

    if bbox_1_area > bbox_2_area:
        mid_height = bbox_2[:,0] + (bbox_2[:,2] - bbox_2[:,0])/2
        mid_width = bbox_2[:,1] + (bbox_2[:,3] - bbox_2[:,1])/2
        bbox_2[:,0] = mid_height - (bbox_1_height/2)
        bbox_2[:,2] = mid_height + (bbox_1_height/2)
        bbox_2[:,1] = mid_width - (bbox_1_width/2)
        bbox_2[:,3] = mid_width + (bbox_1_width/2)
        if bbox_2[:,0] < 0:
            bbox_2[:,2] += abs(bbox_2[:,0])
            bbox_2[:,0] += abs(bbox_2[:,0])
        if bbox_2[:,1] < 0:
            bbox_2[:,3] += abs(bbox_2[:,1])
            bbox_2[:,1] += abs(bbox_2[:,1])

    else:
        mid_height = bbox_1[:,0] + (bbox_1[:,2] - bbox_1[:,0])/2
        mid_width = bbox_1[:,1] + (bbox_1[:,3] - bbox_1[:,1])/2
        bbox_1[:,0] = mid_height - (bbox_2_height/2)
        bbox_1[:,2] = mid_height + (bbox_2_height/2)
        bbox_1[:,1] = mid_width - (bbox_2_width/2)
        bbox_1[:,3] = mid_width + (bbox_2_width/2)
    if bbox_1[:,0] < 0:
        bbox_1[:,2] += abs(bbox_1[:,0])
        bbox_1[:,0] += abs(bbox_1[:,0])
    if bbox_1[:,1] < 0:
        bbox_1[:,3] += abs(bbox_1[:,1])
        bbox_1[:,1] += abs(bbox_1[:,1])

    return np.round(bbox_1).astype(int), np.round(bbox_2).astype(int)

def _limit_num(num, min_val, max_val):
    return num