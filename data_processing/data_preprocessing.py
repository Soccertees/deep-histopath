'''Convert the original datasets into the training and inference format'''

import os
from PIL import Image

from data_processing import data_processing_utils as utils
from data_processing.data_conf import DataConfig


def create_training_masks(train_img_dir,
                          label_dir,
                          output_mask_dir,
                          radius,
                          train_img_suffix='*.tif',
                          label_file_suffix='*.csv',
                          has_header=False):
  """Create the training masks based on the training image size and labels.

  Args:
    train_img_dir: directory path for the training images.
    label_dir: directory path for the label csv files.
    output_mask_dir: directory path for output masks.
    radius: radius for the label point at the pixel unit.
    train_img_suffix: image suffix for the training images.
    label_file_suffix: file suffix for the label csv files.
    has_header:
      boolen value indicating whether the label csv file has the header.
  """
  train_img_files = utils.list_files(train_img_dir, train_img_suffix)
  id_img_dict = utils.get_file_id(train_img_files, utils.FILE_ID_RE)
  label_files = utils.list_files(label_dir, label_file_suffix)
  id_label_dict = utils.get_file_id(label_files, utils.FILE_ID_RE)
  if os.path.exists(output_mask_dir):
    raise FileExistsError("{} already exist".format(output_mask_dir))
  os.makedirs(output_mask_dir, exist_ok=False)

  for id in id_label_dict.keys():
    label_file = id_label_dict[id]
    locations = utils.get_locations_from_csv(
      label_file,
      has_header,
      has_prob=False)
    img_file = id_img_dict[id]
    w, h = utils.get_image_size(img_file)
    mask_file_path = os.path.join(output_mask_dir, "{}.tif".format(id))
    mask = utils.create_mask(h, w, locations, radius)
    mask_img = Image.fromarray(mask)
    mask_dir = os.path.dirname(mask_file_path)
    os.makedirs(mask_dir, exist_ok=True)
    mask_img.save(mask_file_path, subsampling=0, quality=100)


def main():
  conf = DataConfig()

  if not os.path.exists(conf.train_mask_dir):
    print("Start to create the train masks.")
    create_training_masks(
      conf.train_img_dir,
      conf.train_label_dir,
      conf.train_mask_dir,
      conf.radius,
      conf.train_img_suffix,
      conf.train_label_suffix,
      conf.train_label_has_header)


if __name__ == '__main__':
  main()
