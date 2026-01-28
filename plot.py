#!/usr/bin/env python3

import matplotlib.pyplot as plt
from math import sin, cos, pi, ceil, floor

W, H = 400, 100

def sinusoidal(deg):
  """
  Calculate the new height and width of the image.

  >>> # Sinusoidal function formula
  >>> y = A * sin(B * (x-C)) + D
  >>>
  >>> # min and max are shifted
  >>> mid, mx = min([W,H]), max([W,H])
  >>> mn, mx = -(max([W,H])-mid*2)-mid, max([W,H])-mid
  >>>
  >>> # most of intermediary formulars is just constants
  >>> A = (mx - mn) / 2
  >>> T = 360
  >>> B = (2 * pi) / T
  >>> D = (mx + mn) / 2
  >>> C = 0
  >>> x = deg
  >>>
  >>> # sine for height, cosine width (nw)
  >>> return abs(A * sin(B * (x-C)) + D) + mid
  """
  mid, mx = min([W,H]), max([W,H])
  mn, mx = -(max([W,H])-mid*2)-mid, max([W,H])-mid
  x1 = ((2 * pi) / 360) * deg
  ampl = (mx - mn) / 2
  nh = abs(ampl * sin(x1)) + mid
  nw = abs(ampl * cos(x1)) + mid
  return nh, nw

X, Y = zip(*[(deg, sinusoidal(deg)) for deg in range(360)])
yh,yw = zip(*Y)

plt.plot(X, yh, label = "height")
plt.plot(X, yw, label = "width")
plt.xlabel("degrees")
plt.ylabel("pixels")
plt.title(f"new heights and widths of {W}x{H} image when rotated")
plt.legend()
plt.show()

