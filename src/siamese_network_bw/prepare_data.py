#!/usr/bin/env python
# Downloads the LFW (Labeled Faces in the Wild) dataset and converts it to an LMDB database for
# Caffe.

import shutil

import numpy as np

from sklearn.datasets import fetch_lfw_pairs
from sklearn.cross_validation import ShuffleSplit

import leveldb

from caffe_pb2 import Datum

import siamese_network_bw.siamese_utils as siamese_utils
from siamese_network_bw.constants import (
  ROOT,
  TRAINING_FILE,
  VALIDATION_FILE,
  TESTING_FILE,
)

# By default there is a strange split between the training and validation sets;
# the training set is 2200 images while the validation set is 6000 images. We'd rather
# have the split roughly the other way (80/20).
def prepare_training_validation_data(train, validation):
  train_pairs = train["lfw_pairs"]
  validation_pairs = validation["lfw_pairs"]
  data = np.concatenate((train_pairs.data, validation_pairs.data))
  target = np.concatenate((train_pairs.target, validation_pairs.target))

  # After the split we should have 6560 training images and 1640 validation images.
  split = ShuffleSplit(n=len(data), train_size=0.8, test_size=0.2)

  for training_set, validation_set in split:
    X_train = data[training_set]
    y_train = target[training_set]

    X_validation = data[validation_set]
    y_validation = target[validation_set]

  train["lfw_pairs"] = {
    "data": X_train,
    "target": y_train
  }
  validation["lfw_pairs"] = {
    "data": X_validation,
    "target": y_validation
  }

  return (train, validation)

def generate_leveldb(file_path, lfw_pairs, channels, width, height):
  print "Generating LevelDB file at %s..." % file_path
  shutil.rmtree(file_path, ignore_errors=True)
  db = leveldb.LevelDB(file_path)

  batch = leveldb.WriteBatch()
  # The 'data' entry contains both pairs of images unrolled into linear vector.
  for idx, entry in enumerate(lfw_pairs["data"]):
    # Each image pair is a top level key with a keyname like 00059999, in increasing
    # order starting from 00000000.
    key = siamese_utils.get_key(idx)

    # Mean normalize the pixel vector and make sure the target value is correct.
    entry = siamese_utils.mean_normalize(entry)
    label = siamese_utils.normalize_target(lfw_pairs["target"][idx])

    # Each entry in the leveldb is a Caffe protobuffer "Datum" object containing details.
    datum = Datum()
    # One channel for each image in the pair.
    datum.channels = channels
    datum.height = height
    datum.width = width
    # TODO(neuberg): This still doesn't work! I think I need to reshape it a bit
    # according to https://github.com/BVLC/caffe/issues/745
    # and serialize it into bytes correctly; perhaps use array_to_datum over in caffe.
    datum.float_data.extend(entry.flat)
    datum.label = label
    value = datum.SerializeToString()
    db.Put(key, value)

  db.Write(batch, sync = True)

print "Loading LFW data..."
# Each image is 47 (width) x 62 (height). There are 2200 training images, 6000 validation images,
# and 1000 test images; note though that we change the split between training and validation images
# later.
train = {"file_path": TRAINING_FILE, "lfw_pairs": fetch_lfw_pairs(subset="train")}
validation = {"file_path": VALIDATION_FILE, "lfw_pairs": fetch_lfw_pairs(subset="10_folds")}
testing = {"file_path": TESTING_FILE, "lfw_pairs": fetch_lfw_pairs(subset="test")}

m, channels, height, width = testing["lfw_pairs"].pairs.shape

testing["lfw_pairs"] = {
  "data": testing["lfw_pairs"].data,
  "target": testing["lfw_pairs"].target,
}

print "Shuffling & combining data..."
train, validation = prepare_training_validation_data(train, validation)

for entry in [train, validation, testing]:
  generate_leveldb(entry["file_path"], entry["lfw_pairs"], channels, width, height)

print "Done."
