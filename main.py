#!/usr/bin/env python3

import torch

from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F
from torchvision.utils import make_grid

from rotate_fit import RandomRotationFit


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
  trans = T.Compose([
    T.Pad(1, fill=(1., 0., 0.)),
    T.Pad(max([H, W])+100, fill=.7),
    T.CenterCrop(max([H, W])+100),
  ])
  tst = torch.arange(0, 360, 360/16)
  grd = make_grid([trans(RandomRotationFit((deg, deg))(I)) for deg in tst], nrow=4, pad_value=.5)
  F.to_pil_image(grd).show()

