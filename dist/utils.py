from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def toNpArray(row):
  """
  Converts an image row in DataFrame to Numpy array
  :param row: A row that contains the image to be converted.
  :return:
  """
  image = row[0]
  height = image.height
  width = image.width
  nChannels = image.nChannels

  return np.ndarray(
    shape=(height, width, nChannels),
    dtype=np.uint8,
    buffer=image.data,
    strides=(width * nChannels, nChannels, 1))