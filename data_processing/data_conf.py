import os


class DataConfig:
  data_root = "/Users/fei/Documents/Github/deep-histopath/data/mitoses"

  train_img_dir = os.path.join(data_root, "mitoses_train_image_data")

  train_img_suffix = '*.tif'

  train_png_img_dir = os.path.join(data_root, "mitoses_train_png_image_data")

  train_png_img_suffix = '*.png'

  train_label_dir = os.path.join(data_root, "mitoses_train_ground_truth")

  train_label_suffix = '*.csv'

  train_label_has_header = False

  train_mask_dir = os.path.join(data_root, "mitoses_train_mask_data")

  train_mask_suffix = "*.png"

  train_csv_dir = os.path.join(data_root, "mitoses_train_ground_truth")

  train_csv_suffix = "*.csv"

  # TODO: figure out what's the best radius
  radius = 16


class ModelConfig:
  optimizer = 'adam'
  metrics = ['accuracy']
  num_tiles_per_img = 128
  tile_height = 128
  tile_width = 128
  batch_size = 16
  output_channels = 1
  epochs = 20
  val_subslits = 5
  steps_per_epoch = 16
