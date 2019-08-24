import os


class DataConfig:
  data_root = "/Users/fei/Documents/Github/deep-histopath/data/mitoses"

  train_img_dir = os.path.join(data_root, "mitoses_train_image_data")

  train_img_suffix = '*.tif'

  train_label_dir = os.path.join(data_root, "mitoses_train_ground_truth")

  train_label_suffix = '*.csv'

  train_label_has_header = False

  train_mask_dir = os.path.join(data_root, "mitoses_train_mask_data")

  # TODO: figure out what's the best radius
  radius = 16
