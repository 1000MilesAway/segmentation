import torch
from torch import nn
from typing import Any, List

from torch.nn.functional import pad, interpolate
from torch_process import expand_box


class SegmentationPostProcess(torch.nn.Module):
    # def __int__(self):
    #     super(SegmentationPostProcess, self).__init__()
        # # self.expand = torch.jit.script(expand_box)
        # self.mask_proc = torch.jit.script(self.segm_postprocess)

    # @torch.jit.script
    def expand_box(self, box, scale):
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

    # @torch.jit.script
    def segm_postprocess(self, box, raw_cls_mask, im_h, im_w, masks_result):
        raw_cls_mask = pad(raw_cls_mask, (1, 1, 1, 1), 'constant', 0.0)

        scale = torch.tensor(raw_cls_mask.shape[0] / (raw_cls_mask.shape[0] - 2.0))
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

        extended_box = box_exp.to(torch.int)



        inter_size = torch.clamp(extended_box[2:] - extended_box[:2] + 1, min=1).int()
        pt0 = torch.clamp(extended_box[:2],
                             min=torch.zeros(2, dtype=torch.int),
                             max=torch.tensor([im_w, im_h]).to(torch.int))
        pt1 = torch.clamp(extended_box[2:] + 1,
                             min=torch.zeros(2, dtype=torch.int),
                             max=torch.tensor([im_w, im_h]).to(torch.int))

        raw_cls_mask = torch.squeeze(
            interpolate(
                torch.unsqueeze(
                    torch.unsqueeze(raw_cls_mask, dim=0), dim=0), size=(int(inter_size[1]), int(inter_size[0])), mode='bilinear')) > 0.5
        # nupy = raw_cls_mask.numpy()
        mask = raw_cls_mask.to(torch.int)

        # Put an object mask in an image mask.
        masks_result[int(pt0[1]):int(pt1[1]), int(pt0[0]):int(pt1[0])] = mask[(int(pt0[1]) - extended_box[1]):(int(pt1[1]) - extended_box[1]),
                                     (int(pt0[0]) - extended_box[0]):(int(pt1[0]) - extended_box[0])]

    def forward(self, _boxes, _classes, _raw_masks):
        wh = torch.tensor([640, 480])    #x: List[torch.Tensor]  wh: torch.Tensor
        conf = 0.75
        # boxes = x[0][0][:, :4]
        # scores = x[0][0][:, 4]
        # classes = x[1][0]
        # raw_masks = x[2][0]
        boxes = _boxes[0][:, :4]
        scores = _boxes[0][:, 4]
        classes = _classes[0]
        raw_masks = _raw_masks[0]
        detections_filter = scores > conf
        boxes = boxes[detections_filter]
        classes = classes[detections_filter]
        scores = scores[detections_filter]
        raw_masks = raw_masks[detections_filter]
        detections_filter = classes == 0
        boxes = boxes[detections_filter]
        raw_masks = raw_masks[detections_filter]
        scores = scores[detections_filter]
        masks_results = torch.zeros(len(boxes), int(wh[1]), int(wh[0]), dtype=torch.int)
        for box, raw_mask, masks_result in zip(boxes, raw_masks, masks_results):
            # self.mask_proc(box, raw_mask, h, w, masks_result)

            raw_cls_mask = pad(raw_mask, (1, 1, 1, 1), 'constant', 0.0)

            scale = torch.tensor(raw_cls_mask.shape[0] / (raw_cls_mask.shape[0] - 2.0))
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

            extended_box = box_exp.to(torch.int)

            inter_size = torch.clamp(extended_box[2:] - extended_box[:2] + 1, min=1).int()
            pt0 = torch.clamp(extended_box[:2],
                              min=torch.zeros(2, dtype=torch.int),
                              max=torch.tensor([int(wh[0]), int(wh[1])]).to(torch.int))
            pt1 = torch.clamp(extended_box[2:] + 1,
                              min=torch.zeros(2, dtype=torch.int),
                              max=torch.tensor([int(wh[0]), int(wh[1])]).to(torch.int))

            raw_cls_mask = torch.squeeze(
                interpolate(
                    torch.unsqueeze(
                        torch.unsqueeze(raw_cls_mask, dim=0), dim=0), size=(int(inter_size[1]), int(inter_size[0])),
                    mode='bilinear')) > 0.5
            # nupy = raw_cls_mask.numpy()
            mask = raw_cls_mask.to(torch.int)

            # Put an object mask in an image mask.
            masks_result[int(pt0[1]):int(pt1[1]), int(pt0[0]):int(pt1[0])] = mask[(int(pt0[1]) - extended_box[1]):(
                        int(pt1[1]) - extended_box[1]),
                                                                             (int(pt0[0]) - extended_box[0]):(
                                                                                         int(pt1[0]) - extended_box[0])]


        return masks_results, boxes, scores


# postoproc_model = SegmentationPostProcess()

traced_postprocessing = torch.jit.script(SegmentationPostProcess())
torch.jit.save(traced_postprocessing, 'scriptmodule.pt')