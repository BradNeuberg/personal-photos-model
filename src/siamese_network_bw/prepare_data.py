import os
import glob
import random
import shutil
import time

from PIL import Image
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from sklearn.externals import joblib
import plyvel
from caffe_pb2 import Datum

import constants as constants
import siamese_network_bw.siamese_utils as siamese_utils

# We currently restrict ourselves to a subset of the WebFace data. This flag controls
# how many total non-paired faces we pull from this dataset. We currently keep it about the
# same size as LFW.
MAX_FACES = 500

class WebFace:
  """
  Handles loading, pairing, and clustering our data from the WebFace dataset. We currently
  only work with a subset of this dataset for performance reasons.
  """

  def __init__(self):
    # Contains our non-paired images ("data"), along with their correct targets ("target").
    self._train = {
      "data": [],
      "target": [],
    }
    self._validation = {
      "data": [],
      "target": [],
    }
    self._test = {
      "data": [],
      "target": [],
    }

    # Contains training pairs of images ("data"), along with a 1 if they are the same person
    # or a zero if they are not ("target").
    self._train_pairs = {
      "data": [],
      "target": [],
    }

    self._loaded = False

  def is_loaded(self):
    """
    Whether we have loaded our raw, single images and targets into memory. If we haven't,
    load_data() should be called.
    """
    return self._loaded

  def load_data(self):
    """
    Loads our raw face images in and divides them into our training, validation, and
    testing slices.
    """
    print "Loading data..."

    if os.path.isfile(constants.PICKLE_FILE):
      self._load_pickled_data()
      return

    print "\tNo pickled data file available, preparing raw data..."
    identities = self._get_available_identities()
    total_images = 0
    # Keep randomly choosing people until we hit our total number of images to subset.
    while total_images < MAX_FACES:
      person = random.randint(0, len(identities) - 1)
      target_identity = identities[person]
      # Make sure we don't get duplicates for this person.
      del identities[person]

      # Now that we have a person, get all of their images and pre-process them.
      total_images = self._process_images_for_target(target_identity, total_images)

    # Now that we are done, shuffle all of our datasets.
    self._shuffle_all_faces()

    # Now pickle this out to the filesystem.
    self._save_pickled_data()

    self._loaded = True

  def pair_data(self):
    """
    Efficiently pairs every possible positive and negative combination of the faces
    in the training and validation data, persisting them to disk as LevelDB files.
    """
    print "Pairing faces..."

    # Shuffle and pair our faces across the training and validation data sets.
    (train_pairs_data, train_pairs_target) = self._pair_specific_data(
      self._train["data"], self._train["target"], "train")
    (validation_pairs_data, validation_pairs_target) = self._pair_specific_data(
      self._validation["data"], self._validation["target"], "validation")

    # Write them out in batches to our LevelDB files.
    self._generate_leveldb(constants.TRAINING_FILE, train_pairs_data, train_pairs_target,
      self._train["data"])
    self._generate_leveldb(constants.VALIDATION_FILE, validation_pairs_data,
      validation_pairs_target, self._validation["data"])

  def get_clustered_faces(self):
    """
    For several different people, returns images for these along with their targets, for
    both the training and validation data. This makes it easy to test these faces to see
    how they cluster.
    """
    if not self.is_loaded():
      self._load_pickled_data()

    print "Getting clustered faces..."
    # Since the WebFace dataset is fairly balanced, we can simply grab several random faces.
    train_cluster = self._get_cluster_data_for(self._train["data"], self._train["target"])
    validation_cluster = self._get_cluster_data_for(self._validation["data"],
        self._validation["target"])

    print "\t\tTraining cluster, # of samples: %d" % train_cluster["data"].shape[0]
    print "\t\tValidation cluster, # of samples: %d" % validation_cluster["data"].shape[0]

    return {
        "train": train_cluster,
        "validation": validation_cluster,
    }

  def get_validation_images(self):
    """
    Gets our validation data and target details, shaped into being 2D images, loading them if they
    are not loaded yet.
    """
    if not self.is_loaded():
      self._load_pickled_data()

    data = np.reshape(self._validation["data"], (len(self._validation["data"]), 1, constants.WIDTH,
      constants.HEIGHT))
    target = self._validation["target"]

    return {
      "data": data,
      "target": target,
    }

  def _get_cluster_data_for(self, data, target):
    """
    Actually generates cluster data for the given data and target values.
    """
    data_to_keep = []
    target_to_keep = []
    good_identities = []
    identities_to_keep = {}
    for idx in range(len(data)):
      person = target[idx]
      if person in identities_to_keep or len(good_identities) < 5:
        data_to_keep.append(data[idx])
        target_to_keep.append(target[idx])
        if person not in identities_to_keep:
          good_identities.append(person)
          identities_to_keep[person] = True

    data_to_keep = np.array(data_to_keep)
    target_to_keep = np.array(target_to_keep)

    # Scale the data.
    data_to_keep = data_to_keep.reshape(len(data_to_keep), 1, constants.HEIGHT, constants.WIDTH) \
        * 0.00390625
    data_to_keep = self._preprocess_data(data_to_keep)

    return {
      "data": data_to_keep,
      "target": target_to_keep,
      "good_identities": good_identities,
    }

  def _pair_specific_data(self, single_data, single_target, data_name):
    """
    Generates pairs for the specific data and targets passed in.
    """
    print "\tPairing %s data..." % data_name
    pairs_data = []
    pairs_target = []

    # Just work with index references, rather than the actual images, for performance and memory
    # reasons, as there will be too many pairs to fit into memory at once.
    for image_1_idx in range(len(single_data)):
      for image_2_idx in range(len(single_data)):
        pairs_data.append((image_1_idx, image_2_idx))
        same = None
        if single_target[image_1_idx] == single_target[image_2_idx]:
          same = 1
        else:
          same = 0
        pairs_target.append(same)

    # Now shuffle these.
    (pairs_data, pairs_target) = shuffle(pairs_data, pairs_target, random_state=0)

    return (pairs_data, pairs_target)

  def _load_pickled_data(self):
    """
    Load's previously pickled raw non-paired image data and targets from the file system
    so we don't have to pre-process them again. Significantly improves performance.
    """
    print "\tLoading pickled data file..."
    data = joblib.load(constants.PICKLE_FILE)
    self._train = data["train"]
    self._validation = data["validation"]
    self._test = data["test"]
    self._loaded = True

  def _save_pickled_data(self):
    print "\tSaving pickled data..."
    data = {
      "train": self._train,
      "validation": self._validation,
      "test": self._test,
    }
    joblib.dump(data, constants.PICKLE_FILE)

  def _get_available_identities(self):
    """
    Gets the list of unique person identities available in the WebFace dataset.
    """
    identities = []
    for subdir in glob.glob(os.path.join(constants.WEBFACE_DATASET_DIR, "*")):
      subdir = os.path.basename(subdir)
      identities.append(subdir)

    return identities

  def _process_images_for_target(self, target_identity, total_images, max_for_target=100):
    """
    Takes all the face images for a single person, loads them, and pre-processes them such as
    scaling or color conversion. 'max_for_target' controls the maximum number of image we allow
    for this target, to prevent class imbalance.
    """
    print "\t\tProcessing images for target face identity %s" % target_identity
    dir_with_images = os.path.join(constants.WEBFACE_DATASET_DIR, target_identity)
    data = []
    target = []
    total_for_target = 0
    for image_file in glob.glob(os.path.join(dir_with_images, "*.*")):
      total_images = total_images + 1
      im = Image.open(image_file)
      # Note: the CASIA WebFace images are already greyscale.
      # TODO: Crop image instead of resizing based on the coordinates of the two eye centers.
      im.thumbnail((constants.WIDTH, constants.HEIGHT), Image.ANTIALIAS)
      im = np.asarray(im.getdata(), dtype=np.uint8)
      data.append(im)
      target.append(target_identity)
      total_for_target = total_for_target + 1
      if total_for_target >= max_for_target:
        break;

    # Shuffle data into training and validation sets, then further subdivide the validation
    # set into testing data.
    self._shuffle_images_for_target(data, target)

    return total_images

  def _shuffle_images_for_target(self, data, target):
    """
    Takes all the non-paired images for a given person, slices them into training, validation, and
    training sets, and shuffles within each of these sets.
    """
    # train_test_split can only partition into two sets, so we have to partition into two sets, then
    # further partition the validation set into a test set.
    (train_data, other_data, train_target, other_target) = train_test_split(data, target,
      train_size=0.7, test_size=0.3, random_state=0)
    self._train["data"].extend(train_data)
    self._train["target"].extend(train_target)

    (validation_data, test_data, validation_target, test_target) = train_test_split(other_data,
      other_target, train_size=0.9, test_size=0.1, random_state=0)
    self._validation["data"].extend(validation_data)
    self._validation["target"].extend(validation_target)
    self._test["data"].extend(test_data)
    self._test["target"].extend(test_target)

  def _shuffle_all_faces(self):
    """
    Once we have all the non-paired images for the subset of WebFace we are working with, shuffles
    all the values inside each of the major slices for this (i.e. shuffle all the training data,
    shuffle all the validation data, and shuffle all the testing data.)
    """
    print "\tShuffling all faces across all data sets..."
    (train_data, train_target) = shuffle(self._train["data"], self._train["target"], random_state=0)
    self._train["data"] = train_data
    self._train["target"] = train_target

    (validation_data, validation_target) = shuffle(self._validation["data"],
      self._validation["target"], random_state=0)
    self._validation["data"] = validation_data
    self._validation["target"] = validation_target

    (test_data, test_target) = shuffle(self._test["data"], self._test["target"], random_state=0)
    self._test["data"] = test_data
    self._test["target"] = test_target

  def _generate_leveldb(self, file_path, pairs, target, single_data):
    """
    Caffe uses the LevelDB format to efficiently load its training and validation data; this method
    writes paired out faces in an efficient way into this format.
    """
    print "\tGenerating LevelDB file at %s..." % file_path
    shutil.rmtree(file_path, ignore_errors=True)
    db = plyvel.DB(file_path, create_if_missing=True)
    wb = db.write_batch()
    commit_every = 250000
    start_time = int(round(time.time() * 1000))
    for idx in range(len(pairs)):
      # Each image pair is a top level key with a keyname like 00000000011, in increasing
      # order starting from 00000000000.
      key = siamese_utils.get_key(idx)

      # Actually expand our images now, taking the index reference and turning it into real
      # image pairs; we delay doing this until now for efficiency reasons, as we will probably
      # have more pairs of images than actual computer memory.
      image_1 = single_data[pairs[idx][0]]
      image_2 = single_data[pairs[idx][1]]
      paired_image = np.concatenate([image_1, image_2])

      # Do things like mean normalize, etc. that happen across both testing and validation.
      paired_image = self._preprocess_data(paired_image)

      # Each entry in the leveldb is a Caffe protobuffer "Datum" object containing details.
      datum = Datum()
      # One channel for each image in the pair.
      datum.channels = 2 # One channel for each image in the pair.
      datum.height = constants.HEIGHT
      datum.width = constants.WIDTH
      datum.data = paired_image.tostring()
      datum.label = target[idx]
      value = datum.SerializeToString()
      wb.put(key, value)

      if (idx + 1) % commit_every == 0:
        wb.write()
        del wb
        wb = db.write_batch()
        end_time = int(round(time.time() * 1000))
        total_time = end_time - start_time
        print "Wrote batch, key: %s, time for batch: %d ms" % (key, total_time)
        start_time = int(round(time.time() * 1000))

    wb.write()
    db.close()

  def _preprocess_data(self, data):
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
