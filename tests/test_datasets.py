#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring,relative-beyond-top-level
from pathlib import Path

from torch.utils.data import DataLoader
import torchvision

from utils.datasets import YOLOv5
from utils.colors import golden
from utils.labels import COCO80_NAMES
from utils.plots import plot_image


def test_yolov5(plot=False):
  dataset = YOLOv5(Path.home() / 'datasets/coco128', 'train2017')
  print(f'dataset: {len(dataset)}')
  print(f'dataset[0]: {dataset[0]}')
  if plot:
    image, target = dataset[0]
    save_name = 'result.png'
    lb_names = COCO80_NAMES
    lb_colors = [tuple(int(0xff*v) for v in c) for c in golden(len(lb_names))]
    plot_image(image, boxes=target['boxes'], labels=target['labels'],
      lb_names=lb_names, lb_colors=lb_colors, save_name=save_name)
    print(f'draw and save to: {save_name}')


def test_yolov5_loader():
  dataset = YOLOv5(Path.home() / 'datasets/coco128', 'train2017',
    transform=torchvision.transforms.Compose([
      torchvision.transforms.ToTensor()
    ]))

  dataloader = DataLoader(dataset, batch_size=64, shuffle=False,
                          collate_fn=lambda batch: tuple(zip(*batch)))

  for batch_i, (images, targets) in enumerate(dataloader):
    print(f'batch {batch_i}, images {len(images)}, targets {len(targets)}')
    print(f'  images[0]: shape={images[0].shape}')
    print(f'  targets[0]: {targets[0]}')


if __name__ == '__main__':
  test_yolov5()
  print()
  test_yolov5_loader()
