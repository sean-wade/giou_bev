import time
import torch
import numpy as np

from zh_bev_iou.bev_iou_utils import boxes_bev_giou_cpu
from zh_bev_iou.bev_iou_utils import boxes_bev_iou_cpu


a=torch.Tensor([[1,  1,  1,
                 4,  2,  2,
                 0
                ]])

b=torch.Tensor([[5,  1,  1,
                 4,  2,  2,
                 0
                ]])



iou_mat = boxes_bev_iou_cpu(a, b)
print(iou_mat)



giou_mat = boxes_bev_giou_cpu(a, b)
print(giou_mat)

