"""Build the dataset pipeline, generate the UNet model, and train the model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import matplotlib.pyplot as plt

from data_processing.data_processing_utils import list_files, get_file_id
from data_processing.data_conf import DataConfig, ModelConfig
from models.unet.unet_model import unet_model


def tf_read_img(file_path, channels=3):
  img_bytes = tf.io.read_file(file_path)
  img = tf.image.decode_png(img_bytes, channels)
  return img


def tf_convert_to_binary_mask(gray_img):
  mitosis_mask = tf.math.greater(gray_img, 1)
  return mitosis_mask


def tf_gen_coordinates(img_height,
                       img_width,
                       num_point,
                       tile_height=128,
                       tile_width=128):
  offset_height = tf.random.uniform([num_point], minval=0,
                                    maxval=img_height - tile_height,
                                    dtype=tf.dtypes.int32)
  offset_width = tf.random.uniform([num_point], minval=0,
                                   maxval=img_width - tile_width,
                                   dtype=tf.dtypes.int32)
  coordinates = tf.stack([offset_height, offset_width], axis=1)
  return coordinates


def tf_crop_tiles(img, coordinates, dtype, tile_height=128, tile_width=128):
  img_shp = tf.shape(img)
  img_height = img_shp[0]
  img_width = img_shp[1]
  t_h = tf.constant([tile_height], shape=[])
  t_w = tf.constant([tile_width], shape=[])

  def _crop_tile_func(point):
    zero = tf.constant([0], shape=[])
    offset_height = tf.cond(
        point[0] + tile_height >= img_height,
        lambda: tf.maximum(img_height - tile_height, zero),
        lambda: tf.maximum(point[0], zero))
    offset_width = tf.cond(
        point[1] + tile_width >= img_width,
        lambda: tf.maximum(img_width - tile_width, zero),
        lambda: tf.maximum(point[1], zero))

    return tf.image.crop_to_bounding_box(
        img, offset_height, offset_width, t_h, t_w)

  tiles = tf.map_fn(lambda point: _crop_tile_func(point), coordinates, dtype)
  return tiles


def tf_gen_random_point(point, r=16):
  p_0 = tf.random.uniform(
      [], minval=point[0] - r, maxval=point[0] + r, dtype=tf.dtypes.int32)
  p_1 = tf.random.uniform(
      [], minval=point[1] - r, maxval=point[1] + r, dtype=tf.dtypes.int32)
  random_p = tf.stack([p_0, p_1], axis=0)
  return random_p


def tf_gen_augmented_label_points(csv_fp, num_points, radius=16):
  file = tf.io.read_file(csv_fp)
  rows = tf.compat.v1.string_split([file], '\n').values
  label_points = tf.map_fn(
      lambda row: tf.stack(tf.io.decode_csv(row, record_defaults=[-1, -1]),
                           axis=0),
      rows,
      dtype=tf.int32)
  num_label_points = tf.shape(label_points)[0]
  multiplied_label_points = tf.tile(label_points,
                                    [num_points / num_label_points, 1])
  augmented_label_points = tf.map_fn(
      lambda point: tf_gen_random_point(point, radius),
      multiplied_label_points, tf.dtypes.int32)

  return augmented_label_points


def tf_gen_tiles(training_img,
                 mask_img,
                 csv_fp,
                 num_tiles,
                 tile_height=128,
                 tile_width=128):
  shape = tf.shape(training_img)
  img_height = shape[0]
  img_width = shape[1]
  coordinates = tf_gen_coordinates(
      img_height, img_width, num_tiles, tile_height, tile_width)
  # If the training img does not have any mitosis, `csv_fp` will be "None"; For
  # this case, `repeat_augmented_label_points` will be set to be the left-up
  # corner. Otherwise, read the points from the label csv file; then repeat the
  # points to be as same as the random generated tile coordiantes, which keeps
  # the number of mitosis tiles be similar with the normal tiles. Besides, we
  # add a small noise to the coordinates of mitosis points so that the mitois
  # tiles have more differences.
  #
  # TODO: add image augmentation to the tiles.
  repeat_augmented_label_points = tf.cond(
      tf.equal(csv_fp, tf.constant(["None"])),
      lambda: tf.constant([0, 0], shape=[1, 2]),
      lambda: tf_gen_augmented_label_points(csv_fp, num_tiles, radius=16))
  move_dist = tf.constant([int(tile_height/2), int(tile_width/2)],
                          dtype=tf.dtypes.int32)
  # This move may make some coordinates be negative, so in `tf_crop_tiles()`
  # add some logic to handle these edge cases.
  repeat_label_tile_points = repeat_augmented_label_points - move_dist
  tile_coordinates = tf.concat([coordinates, repeat_label_tile_points], axis=0)
  # Shuffle the tile coordinates to mix the normal and mitosis tiles.
  shuffle_coordinates = tf.random.shuffle(tile_coordinates)
  training_tiles = tf_crop_tiles(training_img, shuffle_coordinates,
                                 tf.dtypes.uint8, tile_height, tile_width)
  mask_tiles = tf_crop_tiles(mask_img, shuffle_coordinates, tf.dtypes.bool,
                             tile_height, tile_width)
  return [training_tiles, mask_tiles]


def get_training_datasets(train_img_dir,
                          train_img_suffix,
                          train_mask_dir,
                          train_mask_suffix,
                          train_csv_dir,
                          train_csv_suffix,
                          num_tiles_per_img=128,
                          tile_height=128,
                          tile_width=128):
  train_imgs = list_files(train_img_dir, train_img_suffix)
  train_id_imgs = get_file_id(train_imgs)

  train_masks = list_files(train_mask_dir, train_mask_suffix)
  train_id_masks = get_file_id(train_masks)

  train_csvs = list_files(train_csv_dir, train_csv_suffix)
  train_id_csvs = get_file_id(train_csvs)

  # Some images do not contain any mitosis, so there is no csv files. For these
  # cases, we set the csv file path as 'None'.
  for file_id in train_id_imgs.keys():
    if file_id not in train_id_csvs:
      train_id_csvs[file_id] = "None"

  assert train_id_imgs.keys() == train_id_masks.keys(), \
    "Each training image should have a mask."
  assert train_id_imgs.keys() == train_id_csvs.keys(), \
    "Each training image should have a csv file path."

  fp_for_train_img_mask_csv = [[train_id_imgs[file_id],
                                train_id_masks[file_id],
                                train_id_csvs[file_id]]
                               for file_id in train_id_imgs.keys()]

  dataset = tf.data.Dataset.from_tensor_slices(fp_for_train_img_mask_csv)
  dataset = dataset.shuffle(buffer_size=len(train_imgs))

  dataset = dataset.map(lambda train_img_mask_csv_fp:
                        [tf_read_img(train_img_mask_csv_fp[0], channels=3),
                         tf_convert_to_binary_mask(
                             tf_read_img(train_img_mask_csv_fp[1], channels=1)),
                         train_img_mask_csv_fp[2]],
                        tf.data.experimental.AUTOTUNE)
  dataset = dataset.map(lambda training_img, mask_img, csv_fp:
                        tf_gen_tiles(training_img, mask_img, csv_fp,
                                     num_tiles_per_img, tile_height,
                                     tile_width),
                        tf.data.experimental.AUTOTUNE)
  dataset = dataset.unbatch()
  return dataset


def get_model():
  model_config = ModelConfig()
  model = unet_model(model_config.output_channels)
  model.compile(optimizer=model_config.optimizer,
                loss='binary_crossentropy',
                metrics=model_config.metrics)
  return model


def display(display_list):
  plt.figure(figsize=(30, 30))
  title = ['Input Image', 'True Mask', 'Predicted Mask']
  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    #plt.title(title[i % 3])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()


def examine_model_inputs(num_tiles=32,
                         num_tile_per_row=8,
                         num_tiles_per_img=128,
                         tile_height=128,
                         tile_width=128):
  data_conf = DataConfig()

  dataset = get_training_datasets(
      data_conf.train_png_img_dir,
      data_conf.train_png_img_suffix,
      data_conf.train_mask_dir,
      data_conf.train_mask_suffix,
      data_conf.train_csv_dir,
      data_conf.train_csv_suffix,
      num_tiles_per_img,
      tile_height,
      tile_width)
  sample_imgs = []
  mask_imgs = []
  for image, mask in dataset.take(num_tiles):
    sample_imgs.append(image)
    mask_imgs.append(mask)
    if len(sample_imgs) == num_tile_per_row:
      display(sample_imgs)
      display(mask_imgs)
      sample_imgs.clear()
      mask_imgs.clear()
  display(sample_imgs)
  display(mask_imgs)

def run():
  data_conf = DataConfig()
  model_config = ModelConfig()

  dataset = get_training_datasets(
      data_conf.train_png_img_dir,
      data_conf.train_png_img_suffix,
      data_conf.train_mask_dir,
      data_conf.train_mask_suffix,
      data_conf.train_csv_dir,
      data_conf.train_csv_suffix,
      model_config.num_tiles_per_img,
      model_config.tile_height,
      model_config.tile_width)
  dataset = dataset.batch(model_config.batch_size).repeat()

  model = get_model()
  model.fit(dataset,
            epochs=model_config.epochs,
            steps_per_epoch=model_config.steps_per_epoch)


if __name__ == '__main__':
  run()
