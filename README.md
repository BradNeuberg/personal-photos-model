This repo is an experiment. The attempt is to allow someone to train a neural net over their personal photo collection in order to do face detection on the people in those photos. They can then organize the photos by those people into automatic groups.

This problem is characterized by small labeled training sets, as we only want to ask the user to identify a few of the users to train the net. An individual trained photo model will also be kept private and not shared between photo users for privacy reasons.

This is a playing ground for me to experiment with this; it is a work in progress.

Current Experiment:
* Trying to train Caffe model on LFW (Labeled Faces in the Wild) data set, where input is a single LFW image and output is a softmax over all possible labeled classes (i.e. Joe vs. Mary vs. Tom etc.). Once I have initial "hello world" model I will then experiment with its hyperparameters. The theory is that I can then use this trained model to do "transfer learning" over to an individuals personal photo set and small collection of labels.

Prior Experiments and results:
