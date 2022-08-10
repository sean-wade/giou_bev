import time
import torch
import numpy as np

from zh_bev_iou.bev_iou_utils import boxes_bev_giou_cpu
from zh_bev_iou.bev_iou_utils import boxes_bev_iou_cpu

from py_iou import iou2d, giou2d



boxa = np.array([10., 20., 30, 4, 3, 1, 0]).astype(np.float32)
boxb = np.array([10., 20., 31, 4.2, 3, 1, np.pi/2.0]).astype(np.float32)

a=torch.from_numpy(boxa).reshape(-1, 7)
b=torch.from_numpy(boxb).reshape(-1, 7)

iou_mat = boxes_bev_iou_cpu(a, b)
print("iou = ", iou_mat)

giou_mat = boxes_bev_giou_cpu(a, b)
print("giou = ", giou_mat)


print("py iou = ", iou2d(boxa, boxb))
print("py giou = ", giou2d(boxa, boxb))
