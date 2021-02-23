#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring,invalid-name
import colorsys
import random


def golden(n, h=random.random(), s=0.5, v=0.95, cv_color=False):
  if n <= 0:
    return []

  coef = (1 + 5**0.5) / 2

  colors = []
  for _ in range(n):
    h += coef
    h = h - int(h)
    color = colorsys.hsv_to_rgb(h, s, v)
    if cv_color:
      color = tuple(0xff*v for v in color[::-1])
    colors.append(color)

  return colors
