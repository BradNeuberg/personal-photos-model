# Downloads the LFW (Labeled Faces in the Wild) dataset and converts it to an LMDB database for
# Caffe.

import shutil
import random

import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.cross_validation import ShuffleSplit
from sklearn.utils import shuffle as sklearn_shuffle
import leveldb
from caffe_pb2 import Datum

import constants as constants
import siamese_network_bw.siamese_utils as siamese_utils

def prepare_data(write_leveldb=False):
    """
    Loads our training and validation data, shuffles them, pairs them, and optionally writes them
    to LevelDB databases if 'write_leveldb' is True.
    """
    print "Preparing data..."

    # Each image is 47 (width) x 62 (height). There are 13233 images total, which we will split
    # into 80% training and 20% validation.
    print "\tLoading LFW data..."
    people = fetch_lfw_people()
    data = people["data"]
    target = people["target"]

    # TODO: Bisect these into unique faces and ensure that faces don't leak across training and
    # validation sets.

    print "\tShuffling LFW data..."
    (X_train, Y_train, X_validation, Y_validation) = shuffle(data, target)

    # Cluster the data into pairs.
    print "\tPairing off faces..."
    X_train, Y_train = cluster_all_faces("\t\tTraining", X_train, Y_train, boost_size=10)
    X_validation, Y_validation = cluster_all_faces("\t\tValidation", X_validation, Y_validation,
        boost_size=1)

    # TODO: Print out some statistics, like the number of same, number of different, and the
    # ratio for these different data sets. Perhaps make a new statistics.py file for these.

    train = {
        "file_path": constants.TRAINING_FILE,
        "lfw_pairs": {
            "data": X_train,
            "target": Y_train,
        }
    }
    validation = {
        "file_path": constants.VALIDATION_FILE,
        "lfw_pairs": {
            "data": X_validation,
            "target": Y_validation,
        }
    }

    channels = 2 # One channel for each image in the pair.
    for entry in [train, validation]:
        generate_leveldb(entry["file_path"], entry["lfw_pairs"], channels=channels,
            width=constants.WIDTH, height=constants.HEIGHT)

    print "\tDone preparing data."

    return (train, validation)

def shuffle(data, target):
    """
    Shuffles our data and target into training and validation sets.
    """
    split = ShuffleSplit(n=len(data), train_size=0.8, test_size=0.2, random_state=0)

    for training_set, validation_set in split:
        X_train = data[training_set]
        Y_train = target[training_set]

        X_validation = data[validation_set]
        Y_validation = target[validation_set]

    return (X_train, Y_train, X_validation, Y_validation)

def cluster_all_faces(pair_name, X, Y, boost_size):
    """
    Pairs faces with data in X and targets in Y together, 'boosting' the data set by goinging
    through it 'boost_size' times to amplify the amount of data. Returns our boosted data set
    with paired faces and our target values with 1 for the same face and 0 otherwise.
    """
    X_pairs = []
    Y_pairs = []
    num_pairs = len(X)
    for count in range(boost_size * num_pairs):
        print "%s count: %d" % (pair_name, count)
        pair_images(X, Y, X_pairs, Y_pairs)
    return (np.array(X_pairs), np.array(Y_pairs))

def pair_images(X, Y, X_pairs, Y_pairs):
    """
    Given data (X) and targets (Y), randomly pairs two images together and appends it to
    the array X_pairs, adding 1 to Y_pairs if the images are the same and 0 otherwise.
    """
    num_pairs = len(X)
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

def generate_leveldb(file_path, lfw_pairs, channels, width, height):
    print "\tGenerating LevelDB file at %s..." % file_path
    shutil.rmtree(file_path, ignore_errors=True)
    db = leveldb.LevelDB(file_path)

    batch = leveldb.WriteBatch()
    # The 'data' entry contains both pairs of images unrolled into a linear vector.
    for idx, data in enumerate(lfw_pairs["data"]):
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
        datum.label = lfw_pairs["target"][idx]
        value = datum.SerializeToString()
        db.Put(key, value)

    db.Write(batch, sync = True)

def preprocess_data(data):
    """
    Applies any standard preprocessing we might do on data, whether it is during
    training or testing time. 'data' is a numpy array of unrolled pixel vectors with
    two side by side facial images for each entry.
    """
    # Do nothing for now; testing on the MNIST dataset with our pipeline showed that not mean
    # normalizing and storing our data as bytes, not floats, significantly improved clustering
    # and generalization.
    #data = siamese_utils.mean_normalize(data)

    # We don't scale it's values to be between 0 and 1 as our Caffe model will do that.

    return data

def prepare_cluster_data():
    """
    Load individual faces for 5 different people, rather than pairs. These faces are known
    to have 10 or more images in the testing data.
    """
    print "\tLoading testing cluster data..."
    testing = fetch_lfw_people(min_faces_per_person=10)
    data = testing["data"]
    target = testing["target"]
    identities = testing["target_names"]
    data, target = sklearn_shuffle(data, target, random_state=0)

    # Extract five unique face identities we can work with.
    good_identities = []
    for idx in range(len(target)):
        if target[idx] not in good_identities:
            good_identities.append(target[idx])
        if len(good_identities) == 5:
            break

    # Extract just the indexes with the five good faces we want to work with.
    indexes_to_keep = [idx for idx in range(len(target)) if target[idx] in good_identities]
    data_to_keep = []
    target_to_keep = []
    for i in range(len(indexes_to_keep)):
        keep_me_idx = indexes_to_keep[i]
        data_to_keep.append(data[keep_me_idx])
        target_to_keep.append(target[keep_me_idx])

    data_to_keep = np.array(data_to_keep)

    # Scale the data.
    caffe_in = data_to_keep.reshape(len(data_to_keep), 1, constants.HEIGHT, constants.WIDTH) \
        * 0.00390625
    caffe_in = preprocess_data(caffe_in)

    return (caffe_in, target_to_keep, good_identities, identities)
