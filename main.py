#!/usr/bin/env python3

import torch

from torchvision.transforms.v2.functional import InterpolationMode
from torchvision.transforms.v2 import RandomRotation
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F

from PIL import Image
import numpy as np

from math import cos, sin, pi


class RandomRotationFit(RandomRotation):

  def __init__(
    self,
    degrees: Union[numbers.Number, Sequence],
    interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
    expand: bool = False,
    center: Optional[list[float]] = None,
    fill: Union[_FillType, dict[Union[type, str], _FillType]] = 0,
  ) -> None:
    super().__init__(
      degrees=degrees,
      interpolation=interpolation,
      expand=True,
      center=None,
      fill=fill
    )

  def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
    I = inpt
    _, H, W = I.shape
    r = (H if H < W else W)//2
    I = super().transform(I, params)
    F.to_pil_image(I).show()
    deg = params['angle'] * (pi / 180)
    _, h, w = I.shape
    crop = []
    for x,y in [
      (-(W//2)+r, -(H//2)+r),
      (W - r - W//2, H - r - H//2)
    ]:
      x1 = x*cos(deg) - y*sin(deg)
      y1 = x*sin(deg) + y*cos(deg)
      # TODO sometimes plus y and x, sometimes minus
      if len(crop) == 0:
        crop.append(((int(h//2-y1+r)), int(w//2+x1-r)))
      else:
        crop.append(((int(h//2-y1-r)), int(w//2+x1+r)))
    return I[:,crop[1][0]:crop[0][0],crop[0][1]:crop[1][1]]


if __name__ == '__main__':
  H, W = 200, 800
  r = H//2

  # create example image
  I = torch.zeros((3,H,W))
  I[:] = .2
  for offset in (0, W-r-r-1):
    for x in range(W):
      for y in range(H):
        if (x-r)**2 + (y-r)**2 <= r**2:
          I[:,y,x+offset] = 1.
  I[:,r,r] = torch.tensor([1.,0.,0.])
  I[:,r,W-r] = torch.tensor([1.,0.,0.])
  I[:,H//2,W//2] = torch.tensor([1.,0.,0.])
  F.to_pil_image(I).show()
  F.to_pil_image(RandomRotationFit((45,45))(I)).show()

