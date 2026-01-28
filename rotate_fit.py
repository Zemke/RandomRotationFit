#!/usr/bin/env python3

from torchvision.transforms.v2.functional import InterpolationMode
from torchvision.transforms.v2 import RandomRotation

from math import cos, sin, pi


class RandomRotationFit(RandomRotation):
  """Rotate the input by angle and crop with border radius.

  Use like :class:`RandomRotation`. Rotate around center of image is always assumed.
  """

  def __init__(
    self,
    degrees: Union[numbers.Number, Sequence],
    interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
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
    if params['angle'] == 0.:
      return inpt
    I = super().transform(inpt, params)
    if params['angle'] % 90 == 0:
      return I
    _, H, W = inpt.shape
    mid, mx = min([W,H]), max([W,H])
    mn, mx = -(mx-mid*2)-mid, mx-mid
    x1 = ((2 * pi) / 360) * params['angle']
    ampl = (mx - mn) / 2
    nh = abs(ampl * sin(x1)) + mid
    nw = abs(ampl * cos(x1)) + mid
    _, h, w = I.shape
    dh = int(h-nh)//2
    dw = int(w-nw)//2
    return I[:, dh:-dh, dw:-dw]

