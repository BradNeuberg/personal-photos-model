# Downloads the LFW (Labeled Faces in the Wild) dataset and converts it to an LMDB database for
# Caffe.

import shutil
import random as random

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.datasets import (
    fetch_mldata
)
from sklearn.utils import shuffle
import leveldb
from caffe_pb2 import Datum

import constants as constants
import siamese_network_bw.siamese_utils as siamese_utils

def prepare_data():
    print "Preparing data..."
    print "\tLoading MNIST data..."

    # Each image is 28 (width) x 28 (height). There are 70K images total; we split this into
    # 60K training pairs and 10K validation pairs.
    mnist = fetch_mldata("MNIST original")

    print "\tShuffling & combining data..."
    train, validation = prepare_training_validation_data(mnist)

    channels = 2 # One channel for each image in the pair.
    width = constants.WIDTH
    height = constants.HEIGHT
    for entry in [train, validation]:
        generate_leveldb(entry["file_path"], entry["data"], entry["target"], channels, width, height)

    print "\tDone preparing data."

def prepare_training_validation_data(mnist):
    data = mnist["data"]
    target = mnist["target"]
    data, target = shuffle(data, target, random_state=0)

    # Cluster the data into pairs.
    data_pairs = []
    target_pairs = []
    num_pairs = len(data)
    random.seed(0)
    for idx in range(num_pairs):
        image_1_idx = random.randint(0, num_pairs - 1)
        image_1 = data[image_1_idx]
        image_2_idx = random.randint(0, num_pairs - 1)
        image_2 = data[image_2_idx]
        same = None
        if target[image_1_idx] == target[image_2_idx]:
            same = 1
        else:
            same = 0
        image = np.concatenate([image_1, image_2])
        data_pairs.append(image)
        target_pairs.append(same)
    data_pairs = np.array(data_pairs)
    target_pairs = np.array(target_pairs)

    # Split them into our training and validation sets.
    X_train, X_validation, y_train, y_validation = train_test_split(data_pairs, target_pairs,
        test_size=10000, random_state=0)

    train = {
        "file_path": constants.TRAINING_FILE,
        "data": X_train,
        "target": y_train,
    }
    validation = {
        "file_path": constants.VALIDATION_FILE,
        "data": X_validation,
        "target": y_validation,
    }

    return (train, validation)

def generate_leveldb(file_path, data, target, channels, width, height):
    print "\tGenerating MNIST LevelDB file at %s..." % file_path
    shutil.rmtree(file_path, ignore_errors=True)
    db = leveldb.LevelDB(file_path)

    batch = leveldb.WriteBatch()
    # The 'data' entry contains both pairs of images unrolled into a linear vector.
    for idx, data in enumerate(data):
        # Each image pair is a top level key with a keyname like 00059999, in increasing
        # order starting from 00000000.
        key = siamese_utils.get_key(idx)
        print "\t\tPreparing key: %s" % key

        # Do things like mean normalize, etc. that happen across both testing and validation.
        data = preprocess_data(data)

        # Each entry in the leveldb is a Caffe protobuffer "Datum" object containing details.
        datum = Datum()
        # One channel for each image in the pair.
        datum.channels = channels
        datum.height = height
        datum.width = width
        # Our pixels are float values, such as -3.370285 after being mean normalized.
        datum.float_data.extend(data.astype(float).flat)
        datum.label = target[idx]
        value = datum.SerializeToString()
        db.Put(key, value)

    db.Write(batch, sync = True)

def preprocess_data(data):
    """
    Applies any standard preprocessing we might do on data, whether it is during
    training or testing time. 'data' is a numpy array of unrolled pixel vectors with
    two side by side facial images for each entry.
    """
    data = siamese_utils.mean_normalize(data)

    # We don't scale it's values to be between 0 and 1 as our Caffe model will do that.

    return data

def prepare_testing_cluster_data():
    print "\tLoading testing cluster data..."
    mnist = fetch_mldata("MNIST original")

    data = mnist["data"]
    target = mnist["target"]
    data, target = shuffle(data, target, random_state=0)
    X_train, X_validation, y_train, y_validation = train_test_split(data, target,
        test_size=10000, random_state=0)

    # Scale and normalize the data.
    caffe_in = X_validation.reshape(10000, 1, 28, 28) * 0.00390625
    caffe_in = preprocess_data(caffe_in)

    return (caffe_in, y_validation)
