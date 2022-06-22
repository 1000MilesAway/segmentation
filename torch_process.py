from typing import Any, List

import numpy as np
import cv2
import torch
from torch import nn
from torch.nn.functional import pad, interpolate

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
    box_exp = torch.zeros(box.shape)
    box_exp[0] = x_c - w_half
    box_exp[2] = x_c + w_half
    box_exp[1] = y_c - h_half
    box_exp[3] = y_c + h_half
    return box_exp


def segm_postprocess(box, raw_cls_mask, im_h, im_w, masks_result):
    # Add zero border to prevent upsampling artifacts on segment borders.
    raw_cls_mask = pad(raw_cls_mask, (1, 1, 1, 1), 'constant', 0)
    # nupy = raw_cls_mask.numpy()
    extended_box = expand_box(box, raw_cls_mask.shape[0] / (raw_cls_mask.shape[0] - 2.0)).to(torch.int)
    w, h = torch.clamp(extended_box[2:] - extended_box[:2] + 1, min=1).int().tolist()
    x0, y0 = torch.clamp(extended_box[:2],
                         min=torch.zeros(2, dtype=torch.int),
                         max=torch.Tensor([im_w, im_h]).to(torch.int))
    x1, y1 = torch.clamp(extended_box[2:] + 1,
                         min=torch.zeros(2, dtype=torch.int),
                         max=torch.Tensor([im_w, im_h]).to(torch.int))

    raw_cls_mask = torch.squeeze(
        interpolate(
            torch.unsqueeze(
                torch.unsqueeze(raw_cls_mask, dim=0), dim=0), size=(h, w), mode='bilinear')) > 0.5
    # nupy = raw_cls_mask.numpy()
    mask = raw_cls_mask.to(torch.int)

    # Put an object mask in an image mask.
    # im_mask = torch.zeros(im_h, im_w, dtype=torch.int)
    masks_result[y0:y1, x0:x1] = mask[(y0 - extended_box[1]):(y1 - extended_box[1]),
                                 (x0 - extended_box[0]):(x1 - extended_box[0])]
    # return im_mask



def postprocess(result: Any, wh):
    """
    Postprocesses segmentation model output.
    Overrides default :class:`modules_server.utils.grpc_api.GrpcAPI` postprocessing.
    :param result: model output
    :return: list of person masks borders.
    :rtype: list[list]
    """
    boxes = torch.Tensor(result[0][0][:, :4])
    scores = torch.Tensor(result[0][0][:, 4])
    classes = torch.Tensor(result[1][0])
    raw_masks = torch.Tensor(result[2][0])
    # bl = raw_masks.size()
    detections_filter = scores > DETECTION_MIN_CONFIDENCE
    boxes = boxes[detections_filter]
    classes = classes[detections_filter]
    raw_masks = raw_masks[detections_filter]
    detections_filter = classes == 0
    boxes = boxes[detections_filter]
    raw_masks = raw_masks[detections_filter]
    # masks = []
    masks_results = torch.zeros(len(boxes), int(wh[1]), int(wh[0]), dtype=torch.int)
    for box, raw_mask, masks_result in zip(boxes, raw_masks, masks_results):
        segm_postprocess(box, raw_mask, int(wh[1]), int(wh[0]), masks_result)
        # masks.append(mask)
    return masks_results
    masks_results = masks_results.numpy().astype(np.uint8)
    persons = [cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(-1, -1))[0][0][::5]
               for mask in masks_results]
    borders = []
    for person in persons:
        borders.append([])
        for point in person:
            borders[-1].append([point[0][0] / w, point[0][1] / h])
    return borders



