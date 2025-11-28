#!/usr/bin/env python3

import torch

from torchvision.transforms.v2.functional import InterpolationMode
from torchvision.transforms.v2 import RandomRotation
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F
from torchvision.utils import make_grid

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
    _, H, W = inpt.shape
    r = (H if H < W else W)//2
    I = super().transform(inpt, params)
    deg = params['angle'] * (pi / 180)
    _, h, w = I.shape
    crop = []
    for x,y in [
      (-(W//2)+r, -(H//2)+r),
      (W - r - W//2, H - r - H//2)
    ]:
      x1 = x*cos(deg) - y*sin(deg)
      y1 = x*sin(deg) + y*cos(deg)
      if len(crop) == 0:
        if params['angle'] <= 90:
          crop.append((h//2-y1+r, w//2+x1-r))
        else:
          crop.append((h//2+y1-r,w//2-x1-r))
      else:
        if params['angle'] <= 90:
          crop.append((h//2-y1-r, w//2+x1+r))
        else:
          crop.append((h//2+y1+r,w//2-x1+r))
    if params['angle'] <= 90:
      return I[:, int(crop[1][0]):int(crop[0][0]), int(crop[0][1]):int(crop[1][1])]
    else:
      return I[:, int(crop[0][0]):int(crop[1][0]), int(crop[0][1]):int(crop[1][1])]


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
  I[:,r-4:r+4,r-4:r+4] = torch.tensor([1.,0.,0.]).repeat_interleave(8*8).reshape((3,8,8))
  I[:,r-4:r+4,W-r-4:W-r+4] = torch.tensor([0.,1.,0.]).repeat_interleave(8*8).reshape((3,8,8))
  I[:,H//2-4:H//2+4,W//2-4:W//2+4] = torch.tensor([1.,0.,0.]).repeat_interleave(8*8).reshape((3,8,8))
  F.to_pil_image(I).show()
  trans = T.Compose([
    T.Pad(1, fill=(1., 0., 0.)),
    T.Pad(max([H, W])+100, fill=.7),
    T.CenterCrop(max([H, W])+100),
  ])
  tst = torch.arange(0, 360, 360/16)
  grd = make_grid([trans(RandomRotationFit((deg, deg))(I)) for deg in tst], nrow=4, pad_value=.5)
  F.to_pil_image(grd).show()

