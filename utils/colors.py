#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring,invalid-name
import colorsys
import random


def golden(n, h=random.random(), s=0.5, v=0.95,
           fn=None, scale=None, shuffle=False):
  if n <= 0:
    return []

  coef = (1 + 5**0.5) / 2

  colors = []
  for _ in range(n):
    h += coef
    h = h - int(h)
    color = colorsys.hsv_to_rgb(h, s, v)
    if scale is not None:
      color = tuple(scale*v for v in color)
    if fn is not None:
      color = tuple(fn(v) for v in color)
    colors.append(color)

  if shuffle:
    random.shuffle(colors)
  return colors
