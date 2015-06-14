# Utilities to visualize our facial image data sets.

import math

from sklearn.datasets import fetch_lfw_people
from PIL import Image

import constants


def visualize():
  """
  Writes out various visualizations of our testing data."
  """
  print "Preparing visualizations..."

  tile_faces(fetch_lfw_people()["images"], constants.LOG_DIR + "/all_faces_tiled.png")

def tile_faces(face_data, path):
    """
    Tiles the given faces as an image and writes it to path.
    """
    print "\tTiling faces..."
    num_images_horiz = 10
    num_images_vert = int(math.ceil(len(face_data) / num_images_horiz))
    output_size = (num_images_horiz * constants.WIDTH, num_images_vert * constants.HEIGHT)
    output_img = Image.new("L", output_size)

    idx = 0
    for y in xrange(0, num_images_vert * constants.HEIGHT, constants.HEIGHT):
        for x in xrange(0, num_images_horiz * constants.WIDTH, constants.WIDTH):
            single_face = Image.fromarray(face_data[idx])
            output_img.paste(single_face, (x, y))
            idx += 1

    output_img.save(path)
    print "\t\tTiled faces saved to %s" % path
