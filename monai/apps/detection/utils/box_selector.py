# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =========================================================================
# Adapted from https://github.com/pytorch/vision/blob/main/torchvision/models/detection/retinanet.py
# which has the following license...
# https://github.com/pytorch/vision/blob/main/LICENSE

# BSD 3-Clause License

# Copyright (c) Soumith Chintala 2016,
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
"""
Part of this script is adapted from
https://github.com/pytorch/vision/blob/main/torchvision/models/detection/retinanet.py
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from monai.data.box_utils import batched_nms, box_iou, clip_boxes_to_image
from monai.transforms.utils_pytorch_numpy_unification import floor_divide


class BoxSelector:
    """
    Box selector which selects the predicted boxes.
    The box selection is performed with the following steps:

    #. For each level, discard boxes with scores less than self.score_thresh.
    #. For each level, keep boxes with top self.topk_candidates_per_level scores.
    #. For the whole image, perform non-maximum suppression (NMS) on boxes, with overlapping threshold nms_thresh.
    #. For the whole image, keep boxes with top self.detections_per_img scores.

    Args:
        apply_sigmoid: whether to apply sigmoid to get scores from classification logits
        score_thresh: no box with scores less than score_thresh will be kept
        topk_candidates_per_level: max number of boxes to keep for each level
        nms_thresh: box overlapping threshold for NMS
        detections_per_img: max number of boxes to keep for each image

    Example:

        .. code-block:: python

            input_param = {
                "apply_sigmoid": True,
                "score_thresh": 0.1,
                "topk_candidates_per_level": 2,
                "nms_thresh": 0.1,
                "detections_per_img": 5,
            }
            box_selector = BoxSelector(**input_param)
            boxes = [torch.randn([3,6]), torch.randn([7,6])]
            logits = [torch.randn([3,3]), torch.randn([7,3])]
            spatial_size = (8,8,8)
            selected_boxes, selected_scores, selected_labels = box_selector.select_boxes_per_image(
                boxes, logits, spatial_size
            )
    """

    def __init__(
        self,
        box_overlap_metric: Callable = box_iou,
        apply_sigmoid: bool = True,
        score_thresh: float = 0.05,
        topk_candidates_per_level: int = 1000,
        nms_thresh: float = 0.5,
        detections_per_img: int = 300,
    ):
        self.box_overlap_metric = box_overlap_metric

        self.apply_sigmoid = apply_sigmoid
        self.score_thresh = score_thresh
        self.topk_candidates_per_level = topk_candidates_per_level
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

    def select_top_score_idx_per_level(self, logits: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Select indices with highest scores.

        The indices selection is performed with the following steps:

        #. If self.apply_sigmoid, get scores by applying sigmoid to logits. Otherwise, use logits as scores.
        #. Discard indices with scores less than self.score_thresh
        #. Keep indices with top self.topk_candidates_per_level scores

        Args:
            logits: predicted classification logits, Tensor sized (N, num_classes)

        Return:
            - topk_idxs: selected M indices, Tensor sized (M, )
            - selected_scores: selected M scores, Tensor sized (M, )
            - selected_labels: selected M labels, Tensor sized (M, )
        """
        # import pdb; pdb.set_trace()
        num_classes = logits.shape[-1]

        # apply sigmoid to classification logits if asked
        if self.apply_sigmoid:
            scores = torch.sigmoid(logits.to(torch.float32)).flatten()  ## torch.Size([4199040])/torch.Size([524880])
        else:
            scores = logits.flatten()

        # remove low scoring boxes
        keep_idxs = scores > self.score_thresh  ## threshold=0.02
        scores = scores[keep_idxs]  ## torch.Size([59])/torch.Size([6])/torch.Size([4])
        flatten_topk_idxs = torch.where(keep_idxs)[0]
        ##tensor([ 167751,  167752,  168111,  168112,  168113,  168115,  206631,  206632,
        #  206633,  206991,  206992,  206993,  206994,  206995,  206996,  489642,
        #  489643,  489644,  489999,  490000,  490001,  490002,  490003,  490004,
        #  490005,  490006,  490007,  490359,  490360,  490361,  490362,  490363,
        #  490364,  528879,  528880,  528881,  528882,  528883,  528884,  528885,
        #  528886,  528887, 1409979, 1409980, 1409981, 2861730, 2861731, 2861732,
        # 2861733, 2861734, 2861735, 2862093, 2862094, 2897217, 2897220, 2897221,
        # 2897577, 2900613, 2900614], device='cuda:0')/
        # tensor([ 64101,  64281, 241490, 241493, 251210, 251213], device='cuda:0')  ## size 6

        # keep only topk scoring predictions
        num_topk = min(self.topk_candidates_per_level, flatten_topk_idxs.size(0)) ##59 / 6 / 4
        selected_scores, idxs = scores.to(torch.float32).topk(
            num_topk
        )  # half precision not implemented for cpu float16
        ## idxs = tensor([23, 21, 22, 20, 38, 18, 36, 37, 19, 26, 35, 33, 34, 32, 24, 25, 31, 30,
        # 41, 10,  9, 49, 48, 29, 46, 45,  6,  3, 11, 28,  2,  7, 39, 17, 50, 27,
        # 40, 13, 15, 16, 12, 47, 14,  4, 58,  1,  0, 51, 54, 52, 57,  8, 44, 53,
        # 56, 43,  5, 55, 42], device='cuda:0') / tensor([1, 0, 5, 4, 3, 2], device='cuda:0')
        ## scores= tensor([0.0304, 0.0314, 0.0722, 0.0836, 0.0318, 0.0217, 0.0849, 0.0716, 0.0268,
        # 0.1974, 0.2056, 0.0817, 0.0384, 0.0462, 0.0330, 0.0454, 0.0401, 0.0639,
        # 0.9705, 0.9470, 0.9841, 0.9922, 0.9878, 0.9950, 0.5176, 0.4659, 0.8860,
        # 0.0624, 0.0736, 0.1270, 0.2747, 0.3998, 0.5234, 0.7217, 0.6689, 0.8311,
        # 0.9645, 0.9504, 0.9767, 0.0688, 0.0509, 0.2167, 0.0212, 0.0225, 0.0247,
        # 0.0972, 0.1114, 0.0375, 0.1350, 0.1571, 0.0626, 0.0288, 0.0275, 0.0241,
        # 0.0278, 0.0214, 0.0239, 0.0273, 0.0317], device='cuda:0') / tensor([0.1132, 0.0614, 0.0316, 0.0273, 0.0224, 0.0207], device='cuda:0')
        flatten_topk_idxs = flatten_topk_idxs[idxs]
        # tensor([ 490004,  490002,  490003,  490001,  528884,  489999,  528882,  528883,
        #  490000,  490007,  528881,  528879,  528880,  490364,  490005,  490006,
        #  490363,  490362,  528887,  206992,  206991, 2861734, 2861733,  490361,
        # 2861731, 2861730,  206631,  168112,  206993,  490360,  168111,  206632,
        #  528885,  489644, 2861735,  490359,  528886,  206995,  489642,  489643,
        #  206994, 2861732,  206996,  168113, 2900614,  167752,  167751, 2862093,
        # 2897220, 2862094, 2900613,  206633, 1409981, 2897217, 2897577, 1409980,
        #  168115, 2897221, 1409979], device='cuda:0')/
        # tensor([ 64281,  64101, 251213, 251210, 241493, 241490], device='cuda:0')

        selected_labels = flatten_topk_idxs % num_classes
        # tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0')

        topk_idxs = floor_divide(flatten_topk_idxs, num_classes)
        # tensor([ 490004,  490002,  490003,  490001,  528884,  489999,  528882,  528883,
        #  490000,  490007,  528881,  528879,  528880,  490364,  490005,  490006,
        #  490363,  490362,  528887,  206992,  206991, 2861734, 2861733,  490361,
        # 2861731, 2861730,  206631,  168112,  206993,  490360,  168111,  206632,
        #  528885,  489644, 2861735,  490359,  528886,  206995,  489642,  489643,
        #  206994, 2861732,  206996,  168113, 2900614,  167752,  167751, 2862093,
        # 2897220, 2862094, 2900613,  206633, 1409981, 2897217, 2897577, 1409980,
        #  168115, 2897221, 1409979], device='cuda:0')

        return topk_idxs, selected_scores, selected_labels  # type: ignore

    def select_boxes_per_image(
        self, boxes_list: list[Tensor], logits_list: list[Tensor], spatial_size: list[int] | tuple[int]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Postprocessing to generate detection result from classification logits and boxes.

        The box selection is performed with the following steps:

        #. For each level, discard boxes with scores less than self.score_thresh.
        #. For each level, keep boxes with top self.topk_candidates_per_level scores.
        #. For the whole image, perform non-maximum suppression (NMS) on boxes, with overlapping threshold nms_thresh.
        #. For the whole image, keep boxes with top self.detections_per_img scores.

        Args:
            boxes_list: list of predicted boxes from a single image,
                each element i is a Tensor sized (N_i, 2*spatial_dims)
            logits_list: list of predicted classification logits from a single image,
                each element i is a Tensor sized (N_i, num_classes)
            spatial_size: spatial size of the image

        Return:
            - selected boxes, Tensor sized (P, 2*spatial_dims)
            - selected_scores, Tensor sized (P, )
            - selected_labels, Tensor sized (P, )
        """

        if len(boxes_list) != len(logits_list):  ## boxes_list[0].shape = torch.Size([4199040, 6])
            raise ValueError(
                "len(boxes_list) should equal to len(logits_list). "
                f"Got len(boxes_list)={len(boxes_list)}, len(logits_list)={len(logits_list)}"
            )

        image_boxes = []
        image_scores = []
        image_labels = []

        boxes_dtype = boxes_list[0].dtype
        logits_dtype = logits_list[0].dtype

        for boxes_per_level, logits_per_level in zip(boxes_list, logits_list):
            ## torch.Size([4199040, 6])
            # select topk boxes for each level
            topk_idxs: Tensor
            topk_idxs, scores_per_level, labels_per_level = self.select_top_score_idx_per_level(logits_per_level)
            boxes_per_level = boxes_per_level[topk_idxs] ## torch.Size([59, 6])

            keep: Tensor
            boxes_per_level, keep = clip_boxes_to_image(  # type: ignore
                boxes_per_level, spatial_size, remove_empty=True
            ) ## torch.Size([59, 6])/ / 
            image_boxes.append(boxes_per_level)
            image_scores.append(scores_per_level[keep])
            image_labels.append(labels_per_level[keep])

        import pdb; pdb.set_trace()
        image_boxes_t: Tensor = torch.cat(image_boxes, dim=0) ##torch.Size([69, 6])   59 + 6 + 4 //3个level的box总数
        image_scores_t: Tensor = torch.cat(image_scores, dim=0)##torch.Size([69])
        image_labels_t: Tensor = torch.cat(image_labels, dim=0)##torch.Size([69])

        # non-maximum suppression on detected boxes from all levels
        keep_t: Tensor = batched_nms(  # type: ignore
            image_boxes_t,
            image_scores_t,
            image_labels_t,
            self.nms_thresh,
            box_overlap_metric=self.box_overlap_metric,
            max_proposals=self.detections_per_img,
        )

        selected_boxes = image_boxes_t[keep_t].to(boxes_dtype)
        selected_scores = image_scores_t[keep_t].to(logits_dtype)
        selected_labels = image_labels_t[keep_t]  ## 最终从69个候选框中选出来了6个框作为最后的结果，当然不同参数的设置得出的结果会有差异，视情况而定。
        ## selected_scores = tensor([0.9951, 0.2056, 0.1571, 0.0487, 0.0278, 0.0247], device='cuda:0',dtype=torch.float16
        # selected_labels = tensor([0, 0, 0, 0, 0, 0], device='cuda:0')
    #     (Pdb) selected_boxes
    #     tensor([[ 45.2188, 256.2500,  25.3750,  53.1562, 264.0000,  29.8750],
    #     [ 15.7734, 131.5000, 232.7500,  21.8125, 137.5000, 236.0000],
    #     [289.7500, 257.5000,  59.8438, 295.7500, 263.2500,  62.9688],
    #     [184.1250, 348.5000, 139.1250, 207.0000, 371.5000, 152.5000],
    #     [294.7500, 219.6250, 197.6250, 298.7500, 223.7500, 200.0000],
    #     [139.6250, 108.2500, 143.3750, 148.2500, 116.8125, 148.7500]],
    #    device='cuda:0', dtype=torch.float16)
        return selected_boxes, selected_scores, selected_labels
