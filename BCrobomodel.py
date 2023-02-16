from mmcv import Config
import mmcv
#import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
from PIL import Image
from mmseg.apis import set_random_seed
#import cv2

cfg = Config.fromfile('configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py')

data_root = 'RUGD'
img_dir = 'RUGD_raw'
ann_dir = 'RUGD_annotations'
# # define class and plaette for better visualization
# classes = ("dirt", "sand", "grass", "tree", "pole", "water", "sky", 
#                 "vehicle", "container/generic-object", "asphalt", "gravel", 
#                 "building", "mulch", "rock-bed", "log", "bicycle", "person", 
#                 "fence", "bush", "sign", "rock", "bridge", "concrete", "picnic-table")
# palette = [[ 108, 64, 20 ], [ 255, 229, 204 ],[ 0, 102, 0 ],[ 0, 255, 0 ],
#                 [ 0, 153, 153 ],[ 0, 128, 255 ],[ 0, 0, 255 ],[ 255, 255, 0 ],[ 255, 0, 127 ],
#                 [ 64, 64, 64 ],[ 255, 128, 0 ],[ 255, 0, 0 ],[ 153, 76, 0 ],[ 102, 102, 0 ],
#                 [ 102, 0, 0 ],[ 0, 255, 128 ],[ 204, 153, 255 ],[ 102, 0, 204 ],[ 255, 153, 204 ],
#                 [ 0, 102, 102 ],[ 153, 204, 255 ],[ 102, 255, 255 ],[ 101, 101, 11 ],[ 114, 85, 47 ] ]
# # for file in mmcv.scandir(osp.join(data_root, ann_dir), suffix='.png'):
# #   seg_map = cv2.imread(osp.join(data_root, ann_dir, file), cv2.IMREAD_COLOR)
# #   seg_img = Image.fromarray(seg_map).convert('P')
# #   seg_img.putpalette(np.array(palette, dtype=np.uint8))
# #   seg_img.save(osp.join(data_root, ann_dir, file))
# # for file in mmcv.scandir(osp.join(data_root, ann_dir), suffix='.regions.txt'):
# #   seg_map = np.loadtxt(osp.join(data_root, ann_dir, file)).astype(np.uint8)
# #   seg_img = Image.fromarray(seg_map).convert('P')
# #   seg_img.putpalette(np.array(palette, dtype=np.uint8))
# #   seg_img.save(osp.join(data_root, ann_dir, file.replace('.regions.txt', 
# #                                                          '.png')))
# # Let's take a look at the segmentation map we got
# import matplotlib.patches as mpatches
# # img = Image.open('RUGD/RUGD_annotations/creek_00001.png')
# # plt.figure(figsize=(8, 6))
# # im = plt.imshow(np.array(img.convert('RGB')))

# # # create a patch (proxy artist) for every color 
# # patches = [mpatches.Patch(color=np.array(palette[i])/255., 
# #                           label=classes[i]) for i in range(24)]
# # # put those patched as legend-handles into the legend
# # plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., 
# #            fontsize='large')

# #plt.show()

# split train/val set randomly
split_dir = 'splits'
mmcv.mkdir_or_exist(osp.join(data_root, split_dir))
filename_list = [osp.splitext(filename)[0] for filename in mmcv.scandir(
    osp.join(data_root, ann_dir), suffix='.png')]
with open(osp.join(data_root, split_dir, 'train.txt'), 'w') as f:
  # select first 4/5 as train set
  train_length = int(len(filename_list)*4/5)
  f.writelines(line + '\n' for line in filename_list[:train_length])


with open(osp.join(data_root, split_dir, 'val.txt'), 'w') as f:
  # select last 1/5 as train set
  val_length = int(len(filename_list)*9/10)
  f.writelines(line + '\n' for line in filename_list[train_length:val_length])

with open(osp.join(data_root, split_dir, 'test.txt'), 'w') as f:
  # select last 1/5 as train set

  f.writelines(line + '\n' for line in filename_list[val_length:])

# import os
# file1 = open("/home/hayashi/mmsegmentation/RUGD/train.txt","w")
# dir="/home/hayashi/mmsegmentation/RUGD/RUGD_raw"
# # number_files = len("/home/hayashi/zwy/Rellis-3D/00006/os1_cloud_node_kitti_bin")
# number_files = len([1 for x in list(os.scandir(dir)) if x.is_file()])
# cnt=0
# print(number_files)
# for i, x in enumerate(range(number_files)):
#     file1.write("00010/os1_cloud_node_kitti_bin/{:06d}.bin 00010/os1_cloud_node_semantickitti_label_id/{:06d}.label".format(i ,i))
#     file1.write("\n")


# from mmseg.datasets.builder import DATASETS
# from mmseg.datasets.custom import CustomDataset

# @DATASETS.register_module()
# class RUGDDataset(CustomDataset):
#   CLASSES = classes
#   PALETTE = palette
#   def __init__(self, split, **kwargs):
#     super().__init__(img_suffix='.png', seg_map_suffix='.png', 
#                      split=split, **kwargs)
#     assert osp.exists(self.img_dir) and self.split is not None

# Since we use ony one GPU, BN is used instead of SyncBN
# cfg.norm_cfg = dict(type='BN', requires_grad=True)
# cfg.model.backbone.norm_cfg = cfg.norm_cfg
# cfg.model.decode_head.norm_cfg = cfg.norm_cfg
# cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
# # modify num classes of the model in decode/auxiliary head
# cfg.model.decode_head.num_classes = 24
# cfg.model.auxiliary_head.num_classes = 24

# # Modify dataset type and path
# cfg.dataset_type = 'RUGDDataset'
# cfg.data_root = 'RUGD'

# cfg.data.samples_per_gpu = 2
# cfg.data.workers_per_gpu=2

# cfg.img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# cfg.crop_size = (300, 375)
# cfg.train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations'),
#     dict(type='Resize', img_scale=(688, 550), ratio_range=(0.5, 2.0)),
#     dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='PhotoMetricDistortion'),
#     dict(type='Normalize', **cfg.img_norm_cfg),
#     dict(type='Pad', size=cfg.crop_size, pad_val=0, seg_pad_val=255),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_semantic_seg']),
# ]

# cfg.test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(688, 550),
#         # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **cfg.img_norm_cfg),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]


# cfg.data.train.type = cfg.dataset_type
# cfg.data.train.data_root = cfg.data_root
# cfg.data.train.img_dir = img_dir
# cfg.data.train.ann_dir = ann_dir
# cfg.data.train.pipeline = cfg.train_pipeline
# cfg.data.train.split = 'splits/train.txt'

# cfg.data.val.type = cfg.dataset_type
# cfg.data.val.data_root = cfg.data_root
# cfg.data.val.img_dir = img_dir
# cfg.data.val.ann_dir = ann_dir
# cfg.data.val.pipeline = cfg.test_pipeline
# cfg.data.val.split = 'splits/val.txt'

# cfg.data.test.type = cfg.dataset_type
# cfg.data.test.data_root = cfg.data_root
# cfg.data.test.img_dir = img_dir
# cfg.data.test.ann_dir = ann_dir
# cfg.data.test.pipeline = cfg.test_pipeline
# cfg.data.test.split = 'splits/val.txt'

# # We can still use the pre-trained Mask RCNN model though we do not need to
# # use the mask branch
# # cfg.load_from = 'checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# # Set up working dir to save files and logs.
# cfg.work_dir = './work_dirs/tutorial'

# cfg.runner.max_iters = 200
# cfg.log_config.interval = 10
# cfg.evaluation.interval = 200
# cfg.checkpoint_config.interval = 200

# # Set seed to facitate reproducing the result
# cfg.seed = 0
# set_random_seed(0, deterministic=False)
# cfg.gpu_ids = range(1)

# # Let's have a look at the final config used for training
# print(f'Config:\n{cfg.pretty_text}')

# from mmseg.datasets import build_dataset
# from mmseg.models import build_segmentor
# from mmseg.apis import train_segmentor


# # Build the dataset
# datasets = [build_dataset(cfg.data.train)]

# # Build the detector
# model = build_segmentor(
#     cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
# # Add an attribute for visualization convenience
# model.CLASSES = datasets[0].CLASSES

# # Create work_dir
# mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
# train_segmentor(model, datasets, cfg, distributed=False, validate=True, 
#                 meta=dict())