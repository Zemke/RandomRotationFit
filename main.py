#!/usr/bin/env python3

import torch

from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F
from torchvision.utils import make_grid

from rotate_fit import RandomRotationFit

def marker(t):
  return torch.tensor(t).repeat_interleave(8*8).reshape((3,8,8))

if __name__ == '__main__':
  # exmaple image dimensions
  H, W = 100, 400
  r = (H if H < W else W) // 2

  # create example image
  I = torch.zeros((3,H,W))
  I[:] = .4
  for offset in (0, W-r-r-1):
    for x in range(W):
      for y in range(H):
        if (x-r)**2 + (y-r)**2 <= r**2:
          I[:,y,x+offset] = 1.
  I[:,r-4:r+4,r-4:r+4] = marker([1.,0.,0.])
  I[:,r-4:r+4,W-r-4:W-r+4] = marker([0.,1.,0.])
  I[:,H//2-4:H//2+4,W//2-4:W//2+4] = marker([0.,0.,1.])

  # previews
  trans = T.Compose([
    T.Pad(1, fill=(1., 0., 0.)),
    T.Pad(max([H, W])+r//2, fill=.7),
    T.CenterCrop(max([H, W])+10),
  ])
  tst = torch.arange(0, 360, 360/16)
  for rot in [
    lambda deg: RandomRotationFit((deg, deg)),
    lambda deg: T.RandomRotation((deg, deg)),
    lambda deg: T.RandomRotation((deg, deg), expand=True)
  ]:
    grd = make_grid([trans(rot(deg)(I)) for deg in tst], nrow=4, pad_value=.5)
    F.to_pil_image(grd).show()


