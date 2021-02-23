#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring,import-outside-toplevel,line-too-long
from typing import Union, Optional, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image


def plot_image(
  image: Union[torch.Tensor, Image.Image, np.ndarray],
  boxes: Optional[torch.Tensor] = None,
  labels: Optional[torch.Tensor] = None,
  lb_names: Optional[List[str]] = None,
  lb_colors: Optional[List[Union[str, Tuple[int, int, int]]]] = None,
  save_name: Optional[str] = None,
  show_name: Optional[str] = 'result',
) -> torch.Tensor:
  """
  Draws bounding boxes on given image.
  Args:
    image (Image): `Tensor`, `PIL Image` or `numpy.ndarray`.
    boxes (Optional[Tensor]): `FloatTensor[N, 4]`, the boxes in `[x1, y1, x2, y2]` format.
    labels (Optional[Tensor]): `Int64Tensor[N]`, the class label index for each box.
    lb_names (Optional[List[str]]): All class label names.
    lb_colors (List[Union[str, Tuple[int, int, int]]]): List containing the colors of all class label names.
    save_name (Optional[str]): Save image name.
    show_name (Optional[str]): Show window name.
  """
  if not isinstance(image, torch.Tensor):
    image = torchvision.transforms.ToTensor()(image)

  # torchvision >= 0.9.0/nightly
  #  https://github.com/pytorch/vision/blob/master/torchvision/utils.py
  if boxes is not None:
    if image.dtype != torch.uint8:
      image = torchvision.transforms.ConvertImageDtype(torch.uint8)(image)
    res = torchvision.utils.draw_bounding_boxes(image, boxes,
      labels=[lb_names[i] for i in labels] if labels is not None else None,
      colors=[lb_colors[i] for i in labels] if labels is not None else None)
  else:
    res = image

  if save_name or show_name:
    res = res.permute(1, 2, 0).contiguous().numpy()
    if save_name:
      Image.fromarray(res).save(save_name)
    if show_name:
      plt.gcf().canvas.set_window_title(show_name)
      plt.imshow(res)
      plt.show()

  return res
