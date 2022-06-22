from typing import Any, List

import numpy as np
import cv2


DETECTION_MIN_CONFIDENCE = 0.75
MEAN = np.array([123.675, 116.28, 103.53])
STD = np.array([58.395, 57.12, 57.375])


def expand_box(box, scale):
    w_half = (box[2] - box[0]) * .5
    h_half = (box[3] - box[1]) * .5
    x_c = (box[2] + box[0]) * .5
    y_c = (box[3] + box[1]) * .5
    w_half *= scale
    h_half *= scale
    box_exp = np.zeros(box.shape)
    box_exp[0] = x_c - w_half
    box_exp[2] = x_c + w_half
    box_exp[1] = y_c - h_half
    box_exp[3] = y_c + h_half
    return box_exp


def segm_postprocess(box, raw_cls_mask, im_h, im_w):
    # Add zero border to prevent upsampling artifacts on segment borders.
    raw_cls_mask = np.pad(raw_cls_mask, ((1, 1), (1, 1)), 'constant', constant_values=0)
    extended_box = expand_box(box, raw_cls_mask.shape[0] / (raw_cls_mask.shape[0] - 2.0)).astype(int)
    bl = extended_box[2:] - extended_box[:2] + 1
    w, h = np.maximum(extended_box[2:] - extended_box[:2] + 1, 1)
    x0, y0 = np.clip(extended_box[:2], a_min=0, a_max=[im_w, im_h])
    x1, y1 = np.clip(extended_box[2:] + 1, a_min=0, a_max=[im_w, im_h])

    raw_cls_mask = cv2.resize(raw_cls_mask, (w, h)) > 0.5
    mask = raw_cls_mask.astype(np.uint8)
    # Put an object mask in an image mask.
    im_mask = np.zeros((im_h, im_w), dtype=np.uint8)
    im_mask[y0:y1, x0:x1] = mask[(y0 - extended_box[1]):(y1 - extended_box[1]),
                                 (x0 - extended_box[0]):(x1 - extended_box[0])]
    return im_mask


def preprocess(image: np.ndarray, width, height) -> np.ndarray:
    image = cv2.resize(image, (width, height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.float32(image)
    image = (image - MEAN) / STD
    image = image.transpose(2, 0, 1)
    return image

def postprocess(result: Any) -> List[List]:
    """
    Postprocesses segmentation model output.
    Overrides default :class:`modules_server.utils.grpc_api.GrpcAPI` postprocessing.
    :param result: model output
    :return: list of person masks borders.
    :rtype: list[list]
    """
    boxes = result[0][0][:, :4]
    scores = result[0][0][:, 4]
    classes = result[1][0]
    raw_masks = result[2][0]
    detections_filter = scores > DETECTION_MIN_CONFIDENCE
    boxes = boxes[detections_filter]
    classes = classes[detections_filter]
    raw_masks = raw_masks[detections_filter]
    detections_filter = classes == 0
    boxes = boxes[detections_filter]
    raw_masks = raw_masks[detections_filter]
    masks = []
    for box, raw_mask in zip(boxes, raw_masks):
        mask = segm_postprocess(box, raw_mask, 800, 1440)
        masks.append(mask)
    persons = [cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(-1, -1))[0][0][::5]
               for mask in masks]
    borders = []
    for person in persons:
        borders.append([])
        for point in person:
            borders[-1].append([point[0][0] / 1440, point[0][1] / 800])
    return borders