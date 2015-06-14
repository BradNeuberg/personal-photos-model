#!/usr/bin/env python
import argparse
import os

import constants as constants
from prepare_data import prepare_data
from train import train
from predict import predict

def parse_command_line():
    parser = argparse.ArgumentParser(
        description="""Train, validate, and test a face detection classifier that will determine if
        two faces are the same or different.""")
    parser.add_argument("-p", "--prepare-data", help="Prepare training and validation data.",
        action="store_true")
    parser.add_argument("-t", "--train", help="""Train classifier. Use --graph to generate quality
        graphs""", action="store_true")
    parser.add_argument("-g", "--graph", help="Generate training graphs.", action="store_true")
    parser.add_argument("--weights", help="The trained model weights to use; if not provided " +
        "defaults to " + constants.TRAINED_WEIGHTS, type=str, default=constants.TRAINED_WEIGHTS)
    parser.add_argument("--note", help="Adds extra note onto generated quality graph.", type=str)
    parser.add_argument("-s", "--is_same", help="""Determines if the two images provided are the
        same or different. Provide relative paths to both images.""", nargs=2, type=str)
    args = vars(parser.parse_args())

    if os.environ.get("CAFFE_HOME") == None:
        print "You must set CAFFE_HOME to point to where Caffe is installed. Example:"
        print "export CAFFE_HOME=/usr/local/caffe"
        exit(1)

    if args["prepare_data"] == True:
        prepare_data()
    if args["train"] == True:
        train(args["graph"], weight_file=args["weights"], note=args["note"])
    if args["is_same"] != None:
        # TODO: Fill this out once we have a threshold and neural network trained.
        images = args["is_same"]
        predict(images[0], images[1])

if __name__ == "__main__":
    parse_command_line()
