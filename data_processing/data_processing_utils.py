'''Utils functions for data preprocessing'''
import numpy as np
import os
import pandas as pd
from pathlib import Path
from PIL import Image
import re
import sys


FILE_ID_RE = "(\d+/\d+|\d+-\d+)[.csv|/clustered_mitosis_locations.csv|.tif|_mark.tif|_mask.tif|.png|_mark.png|_mask.png]"

def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()

def get_image_size(img_file):
  """ Get the size of the input image file.

  Args:
    img_file: input image file path.

  Returns:
    [w, h]: the width and height of the input image
  """
  with Image.open(img_file) as img:
    w, h = img.size
    return w, h


def create_mask(h, w, coords, radius):
  """Create a binary image mask with locations of mitosis patches.

  Pixels equal to one indicate normal regions, while areas equal to one
  indicate mitosis regions.  More specifically, all locations within a
  Euclidean distance <= `radius` from the center of a true mitosis are
  set to a value of one, and all other locations are set to a value of
  zero.

  Args:
    h: Integer height of the mask.
    w: Integer width of the mask.
    coords: A list-like collection of (row, col) mitosis coordinates.
    radius: An integer radius of the circular patches to place on the
      mask for each mitosis location.

  Returns:
    A binary mask of the same shape as `im` indicating where the
    mitosis patches are located.
  """
  # check that row, col, and size are within the image bounds
  # assert 1 < size <= min(h, w), "size must be >1 and within the bounds of the image"

  # create mitosis patch mask
  mask = np.zeros((h, w), dtype=np.bool)
  for row, col in coords:
    assert 0 <= row <= h, "row is outside of the image height"
    assert 0 <= col <= w, "col is outside of the image width"

    # mitosis mask as a circle with radius `radius` pixels centered on the given location
    y, x = np.ogrid[:h, :w]
    mitosis_mask = np.sqrt((y - row) ** 2 + (x - col) ** 2) <= radius

    # indicate mitosis patch area on mask
    mask = np.logical_or(mask, mitosis_mask)

  return mask


def list_files(file_dir, file_suffix):
  """ recursively list all the files that have the same input file
    suffix under the input directory.

  Args
    dir: file directory.
    file_suffix: file suffix, e.g. '*.tif'.

  Return:
    a list of file path.
  """
  dir_path = Path(file_dir)
  files = [str(x) for x in dir_path.rglob(file_suffix)]
  return files


def get_file_id(files, file_id_re=FILE_ID_RE):
  """ get the file id using the file id regular expression
  Args:
    files: list of input file paths
    file_id_re: regular expression string used to detect the file ID
      from the file path

  Return:
    a dictionary of file id and its full file path
  """
  id_files = {re.findall(file_id_re, x)[0]: x for x in files}
  return id_files


def get_locations_from_csv(file, has_header=False, has_prob=True):
  """ get the point locations from CSV file.

  Args:
    file: csv file, of which the first and second columns store the
      point coordinates.
    has_header: bool value to tell if the input csv file has a header or
      not.
    has_prob: bool value to tell if the input csv file has the
      probability column

  Return:
    a list of point locations, e.g. [(r0, c0, p0), (r1, c1, p1), ......].
  """
  # handle the case that the input file does not exist
  if file is None:
    return []

  if has_header:
    data = pd.read_csv(file)
  else:
    data = pd.read_csv(file, header=None)

  if has_prob:
    locations = [(int(x[0]), int(x[1]), float(x[2])) for x in
                 data.values.tolist()]
  else:
    locations = [(int(x[0]), int(x[1])) for x in data.values.tolist()]
  return locations
