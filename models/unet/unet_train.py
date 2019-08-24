from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.unet.model.model import Unet

model = Unet(
  backbone_name="resnet50",
  encoder_weights="imagenet",
  decoder_block_type="transpose")

model.compile('Adam', 'binary_crossentropy', ['binary_accuracy'])