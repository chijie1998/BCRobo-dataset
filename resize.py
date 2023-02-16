import mmcv
import matplotlib.pyplot as plt
import cv2
import os.path as osp
import numpy as np
from PIL import Image
# convert dataset annotation to semantic segmentation map
data_root = 'RUGD'
img_dir = 'RUGD_raw'
ann_dir = 'RUGD_annotations'
          
for file in mmcv.scandir(osp.join(data_root, img_dir), suffix='.jpg'):           
  image = Image.open(osp.join(data_root, img_dir, file))
  file1,fileext= osp.splitext(file)
  # image = image.resize((688, 550))
  image.save(osp.join(data_root, img_dir,file1 + ".png"),"PNG")

# i=cv2.imread('RUGD/RUGD_annotations/creek_00001.png')
# cv2.imshow("image",i)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# split train/val set randomly
# split_dir = 'splits'
# mmcv.mkdir_or_exist(osp.join(data_root, split_dir))
# filename_list = [osp.splitext(filename)[0] for filename in mmcv.scandir(
#     osp.join(data_root, ann_dir), suffix='.png')]
# with open(osp.join(data_root, split_dir, 'train.txt'), 'w') as f:
#   # select first 4/5 as train set
#   train_length = int(len(filename_list)*4/5)
#   f.writelines(line + '\n' for line in filename_list[:train_length])
# with open(osp.join(data_root, split_dir, 'val.txt'), 'w') as f:
#   # select last 1/5 as train set
#   f.writelines(line + '\n' for line in filename_list[train_length:])

# from mmseg.datasets.builder import DATASETS
# from mmseg.datasets.custom import CustomDataset

# @DATASETS.register_module()
# class RUGDDataset(CustomDataset):
#   CLASSES = classes
#   PALETTE = palette
#   def __init__(self, split, **kwargs):
#     super().__init__(img_suffix='.jpg', seg_map_suffix='.png', 
#                      split=split, **kwargs)
#     assert osp.exists(self.img_dir) and self.split is not None