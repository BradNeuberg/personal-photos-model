# Downloads the LFW (Labeled Faces in the Wild) dataset and converts it to an LMDB database for
# Caffe.

import shutil
import random
import glob

from PIL import Image
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.cross_validation import ShuffleSplit
from sklearn.utils import shuffle as sklearn_shuffle
from sklearn.externals import joblib
import leveldb
from caffe_pb2 import Datum

import constants as constants
import siamese_network_bw.siamese_utils as siamese_utils

def prepare_data(write_leveldb=True, pair_faces=True, use_pickle=False):
    """
    Loads our training and validation data, shuffles them, pairs them if 'pair_faces' is True, and
    optionally writes them to LevelDB databases if 'write_leveldb' is True. If 'use_pickle' is
    True, we load our data set from a previously pickled set of faces.
    """
    print "Preparing data..."

    if use_pickle == True:
        print "\tLoading pickled LFW data..."
        return joblib.load(constants.PICKLE_FILE)

    # Each image is 58 (width) x 58 (height). There are 13233 images total, which we will split
    # into 80% training and 20% validation.
    print "\tLoading LFW data..."
    (data, target) = load_lfw()

    # Disable this for now.
    #print "\tFiltering faces for consistent counts..."
    (data, target) = ensure_face_count(data, target)

    # TODO: Bisect these into unique faces and ensure that faces don't leak across training and
    # validation sets.

    print "\tShuffling LFW data..."
    (X_train, y_train, X_validation, y_validation) = shuffle(data, target)

    X_train_pairs = None
    y_train_pairs = None
    X_validation_pairs = None
    y_validation_pairs = None
    if pair_faces == True:
        # Cluster the data into pairs.
        print "\tPairing off training faces..."
        X_train_pairs, y_train_pairs = cluster_all_faces("\t\tTraining", X_train, y_train,
            boost_size=1)
        print "\tPairing off validation faces..."
        X_validation_pairs, y_validation_pairs = cluster_all_faces("\t\tValidation", X_validation,
            y_validation, boost_size=20)
        print "\tFinished pairing!"

    # TODO: Print out some statistics, like the number of same, number of different, and the
    # ratio for these different data sets. Perhaps make a new statistics.py file for these.

    train = {
        "file_path": constants.TRAINING_FILE,
        "lfw": {
            "data": X_train,
            "target": y_train,
        },
        "lfw_pairs": {
            "data": X_train_pairs,
            "target": y_train_pairs,
        },
    }
    validation = {
        "file_path": constants.VALIDATION_FILE,
        "lfw": {
            "data": X_validation,
            "target": y_validation,
        },
        "lfw_pairs": {
            "data": X_validation_pairs,
            "target": y_validation_pairs,
        },
    }

    if write_leveldb == True:
        channels = 2 # One channel for each image in the pair.
        for entry in [train, validation]:
            print "\t\tWriting leveldb database..."
            generate_leveldb(entry["file_path"], entry["lfw_pairs"], channels=channels,
                width=constants.WIDTH, height=constants.HEIGHT)

    print "\tSaving and pickling LFW data..."
    joblib.dump((train, validation), constants.PICKLE_FILE)

    print "\tDone preparing data."

    return (train, validation)

def load_lfw():
    """
    Loads and returns our LFW dataset.
    """
    # Don't use Scikit's LFW data for now.
    # people = fetch_lfw_people()
    # data = people["data"]
    # target = people["target"]

    # The directory names are our targets, while individual files inside that directory are the
    # faces for that target.
    data = []
    target = []
    for (target_idx, target_name) in enumerate(glob.glob(constants.LFW_DATASET_DIR + "/*")):
        for image_filename in glob.glob(target_name + "/*"):
            print "\t\tOpening %s" % image_filename
            im = Image.open(image_filename, "r")
            # Convert to greyscale.
            im = im.convert("L")
            # TODO: Crop image to 58x58 instead of resizing based on the coordinates of the two eye
            # centers.
            im.thumbnail((58, 58), Image.ANTIALIAS)
            im = np.asarray(im.getdata(), dtype=np.uint8)
            data.append(im)
            target.append(target_idx)

    return (np.asarray(data), np.asarray(target))

def shuffle(data, target):
    """
    Shuffles our data and target into training and validation sets.
    """
    split = ShuffleSplit(n=len(data), train_size=0.8, test_size=0.2, random_state=0)

    for training_set, validation_set in split:
        X_train = data[training_set]
        y_train = target[training_set]

        X_validation = data[validation_set]
        y_validation = target[validation_set]

    return (X_train, y_train, X_validation, y_validation)

def ensure_face_count(data, target, min_count=100, max_count=2000):
    """
    Goes through faces in the data and ensures that we never have less than
    'min_count' or more than 'max_count'.
    """
    data_results = []
    target_results = []

    # First build up lookup table going from every face number to each image.
    faces = {}
    for idx in xrange(data.shape[0]):
        person = target[idx]
        entry = faces.get(str(person), [])
        entry.append(data[idx])
        faces[str(person)] = entry

    # Now only choose ones that have at least 'min_count' number of entries, and limit those that do
    # to just 'max_count'.
    for key in faces:
        entry = faces[key]
        if len(entry) < min_count:
            continue
        for idx in xrange(len(entry)):
            if idx == max_count:
                break
            data_results.append(entry[idx])
            target_results.append(key)

    print "\t\tFiltered faces to min_count=%d and max_count=%d, leaving a total of %d image faces" \
        % (min_count, max_count, len(data_results))

    return (np.asarray(data_results), np.asarray(target_results))

def cluster_all_faces(pair_name, X, y, boost_size):
    """
    Pairs faces with data in X and targets in y together, 'boosting' the data set by going
    through it 'boost_size' times to amplify the amount of data. Returns our boosted data set
    with paired faces and our target values with 1 for the same face and 0 otherwise.
    """
    X_pairs = []
    y_pairs = []
    num_pairs = len(X)
    print "num_pairs: %d" % num_pairs
    for image_1_idx in range(num_pairs):
        image_1 = X[image_1_idx]
        for image_2_idx in range(num_pairs):
            print "image_1_idx: %d, image_2_idx: %d" % (image_1_idx, image_2_idx)
            image_2 = X[image_2_idx]
            same = None
            if y[image_1_idx] == y[image_2_idx]:
                same = 1
            else:
                same = 0
            image = np.concatenate([image_1, image_2])
            X_pairs.append(image)
            y_pairs.append(same)

    print "Produced X_pairs: %d" % len(X_pairs)

    return (np.array(X_pairs), np.array(y_pairs))

def pair_images(X, y, X_pairs, y_pairs):
    """
    Given data (X) and targets (y), randomly pairs two images together and appends it to
    the array X_pairs, adding 1 to y_pairs if the images are the same and 0 otherwise.
    """
    num_pairs = len(X)
    image_1_idx = random.randint(0, num_pairs - 1)
    image_1 = X[image_1_idx]
    image_2_idx = random.randint(0, num_pairs - 1)
    image_2 = X[image_2_idx]
    same = None
    if y[image_1_idx] == y[image_2_idx]:
        same = 1
    else:
        same = 0
    image = np.concatenate([image_1, image_2])
    X_pairs.append(image)
    y_pairs.append(same)

def generate_leveldb(file_path, lfw_pairs, channels, width, height):
    print "\tGenerating LevelDB file at %s..." % file_path
    shutil.rmtree(file_path, ignore_errors=True)
    db = leveldb.LevelDB(file_path)

    batch = leveldb.WriteBatch()
    commit_every = 500000
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

        if idx % commit_every == 0:
            print "Comitting batch %d..." % (idx/commit_every)
            db.Write(batch, sync=True)
            del batch
            batch = leveldb.WriteBatch()

    db.Write(batch, sync=True)

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
    print "\tPreparing cluster data..."
    (train, validation) = prepare_data(write_leveldb=False, pair_faces=False, use_pickle=True)

    train_cluster = get_cluster_data_for(data=train["lfw"]["data"], target=train["lfw"]["target"])
    validation_cluster = get_cluster_data_for(data=validation["lfw"]["data"],
        target=validation["lfw"]["target"])

    print "\t\tTraining cluster, # of samples: %d" % train_cluster["data"].shape[0]
    print "\t\tValidation cluster, # of samples: %d" % validation_cluster["data"].shape[0]

    return {
        "train": train_cluster,
        "validation": validation_cluster,
    }

def get_cluster_data_for(data, target, min_count=10):
    """
    Actually generates cluster data for the given data and target values.
    """
    # TODO: We might not need this anymore now that we are ensuring we have enough faces.
    # Extract five unique face identities we can work with.
    good_identities = []
    num_hits = {}
    for idx in range(len(target)):
        # How many hits have we seen for this particular face so far?
        identity = str(target[idx])
        hits_for_face = num_hits.get(identity, 0)
        hits_for_face = hits_for_face + 1
        num_hits[identity] = hits_for_face

        # Does this have enough faces for us to care, and have we not seen it before?
        if hits_for_face >= min_count and target[idx] not in good_identities:
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

    return {
        "data": caffe_in,
        "target": target_to_keep,
        "good_identities": good_identities,
    }
