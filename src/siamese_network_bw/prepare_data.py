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

    # Split them into our training and validation sets.
    X_train, X_validation, Y_train, Y_validation = train_test_split(data, target,
        test_size=10000, random_state=0)

    # Cluster the data into pairs.
    X_train_pairs = []
    Y_train_pairs = []
    num_train_pairs = len(X_train)
    random.seed(0)
    for count in range(10 * num_train_pairs):
        print "training count: %d" % count
        pair_images(num_train_pairs, X_train, Y_train, X_train_pairs, Y_train_pairs)
    X_train = np.array(X_train_pairs)
    Y_train = np.array(Y_train_pairs)

    X_validation_pairs = []
    Y_validation_pairs = []
    num_validation_pairs = len(X_validation)
    for count in range(num_validation_pairs):
        print "validation count: %d" % count
        pair_images(num_validation_pairs, X_validation, Y_validation, X_validation_pairs,
            Y_validation_pairs)
    X_validation = np.array(X_validation_pairs)
    Y_validation = np.array(Y_validation_pairs)

    print "# of same training pairs: %d" % np.sum(Y_train)
    print "# of same validation pairs: %d" % np.sum(Y_validation)

    train = {
        "file_path": constants.TRAINING_FILE,
        "data": X_train,
        "target": Y_train,
    }
    validation = {
        "file_path": constants.VALIDATION_FILE,
        "data": X_validation,
        "target": Y_validation,
    }

    return (train, validation)

def pair_images(num_pairs, X, Y, X_pairs, Y_pairs):
    image_1_idx = random.randint(0, num_pairs - 1)
    image_1 = X[image_1_idx]
    image_2_idx = random.randint(0, num_pairs - 1)
    image_2 = X[image_2_idx]
    same = None
    if Y[image_1_idx] == Y[image_2_idx]:
        same = 1
    else:
        same = 0
    image = np.concatenate([image_1, image_2])
    X_pairs.append(image)
    Y_pairs.append(same)

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
        datum.data = data.tobytes()
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
    # NOTE(neuberg): Disable doing any mean normalization for now, as the Caffe MNIST vanilla
    # converter doesn't do this and seems to get good results. Maybe convert this to use a Caffe
    # MVN layer if we still want to do it?
    #data = siamese_utils.mean_normalize(data)

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
