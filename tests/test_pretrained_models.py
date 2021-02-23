#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
import torch
import torchvision
from PIL import Image

from utils.colors import golden
from utils.plots import plot_image


COCO_INSTANCE_CATEGORY_NAMES = [
  '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
  'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
  'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
  'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
  'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
  'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
  'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
  'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
  'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
  'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
  'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
  'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def test_fasterrcnn(image='data/bicycle.jpg', device=None, score=0.9):
  if device is None:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  # model

  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
  print(model)

  model.to(device)
  model.eval()

  # image

  img = Image.open(image).convert("RGB")
  img = torchvision.transforms.ToTensor()(img)
  images = [img.to(device)]

  # inference

  predictions = model(images)
  pred = predictions[0]
  print(pred)

  # plot

  scores = pred['scores']
  mask = scores >= score

  boxes = pred['boxes'][mask]
  labels = pred['labels'][mask]
  scores = scores[mask]

  lb_names = COCO_INSTANCE_CATEGORY_NAMES
  lb_colors = golden(len(lb_names), fn=int, scale=0xff, shuffle=True)
  lb_infos = [f'{s:.2f}' for s in scores]
  plot_image(img, boxes, labels, lb_names, lb_colors, lb_infos,
             save_name='result.png')


if __name__ == '__main__':
  test_fasterrcnn()
