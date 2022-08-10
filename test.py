import time
import torch
import numpy as np

from zh_bev_iou.bev_iou_utils import boxes_bev_giou_cpu
from zh_bev_iou.bev_iou_utils import boxes_bev_iou_cpu


# a=torch.Tensor([[1,1,1,
#                  2,2,2,
#                  3.0]])

# b=torch.Tensor([[1,1,1,
#                  2.1,2,2,
#                  0.1]])

a = torch.rand((5, 7))
b = torch.rand((5, 7))
                 
s = time.time()                
for i in range(1):
    iou_mat = boxes_bev_iou_cpu(a, b)
e = time.time()  
print(" iou cpu using: %.2fs"%(e - s))
print(iou_mat)


s = time.time()                
for i in range(1):
    giou_mat = boxes_bev_giou_cpu(a, b)
e = time.time()  
print("giou cpu using: %.2fs"%(e - s))
print(giou_mat)
