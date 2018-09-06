from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
from pyspark.sql.types import BinaryType, StringType, StructField, StructType
from PIL import Image
from io import BytesIO
import logging


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

def image_decoder(rawbytes):
  """
  Decode raw bytes to image array
  :param rawbytes:
  :return: a numpy array with uint8
  """
  img = Image.open(BytesIO(rawbytes))
  array = np.asarray(img, dtype=np.uint8)
  return array


def genBinaryFileRDD(sc, path, numPartitions=None):
  """
    Read files from a directory to a RDD.
    :param sc: SparkContext.
    :param path: str, path to files.
    :param numPartition: int, number or partitions to use for reading files.
    :return: RDD with a pair of key and value: (filePath: str, fileData: BinaryType)
    """
  numPartitions = numPartitions or sc.defaultParallelism
  rdd = sc.binaryFiles(
    path, minPartitions=numPartitions).repartition(numPartitions)
  #rdd = rdd.map(lambda x: (x[0], bytearray(x[1])))
  return rdd