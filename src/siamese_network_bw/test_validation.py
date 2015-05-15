#!/usr/bin/env python
import caffe

import leveldb

from caffe_pb2 import Datum

import siamese_network_bw.siamese_utils as siamese_utils
from siamese_network_bw.constants import (
  WIDTH,
  HEIGHT,
  VALIDATION_FILE,
  VALIDATION_SIZE,
)

MODEL_FILE = "src/siamese_network_bw/model/siamese.prototxt"
# TODO(neuberg): Move the trained file to a better location and give it a better file name.
PRETRAINED_FILE = "_iter_1501.caffemodel"

def main():
  validation_db = leveldb.LevelDB(VALIDATION_FILE)

  caffe.set_mode_gpu()
  classifier = caffe.Classifier(MODEL_FILE, PRETRAINED_FILE,
        image_dims=(WIDTH, HEIGHT),
        input_scale=1
  )

  for i in xrange(VALIDATION_SIZE):
    key = siamese_utils.get_key(i)
    entry = Datum.FromString(validation_db.Get(key))
    data = entry.data
    score = classifier.predict(entry.data, oversample=False)
    print score

# TODO(neuberg): Figure out why I can't run this from the command-line yet (I have to use ipython).
# It's an Anaconda issue; probably have to recompile Caffe against system python rather than
# Anaconda.
if __name__ == "__main__":
  main()
