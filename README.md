This repo is an experiment. The attempt is to allow someone to train a neural net over their personal photo collection in order to do face detection on the people in those photos. They can then organize the photos by those people into automatic groups.

This problem is characterized by small labeled training sets, as we only want to ask the user to identify a few of the users to train the net. An individual trained photo model will also be kept private and not shared between photo users for privacy reasons, which can make training more challenging.

This is a playing ground for me to experiment with this; it is a work in progress.

Setup:
* Add the lib/ directory to your PYTHONPATH where you installed this. Example:
export PYTHONPATH=$PYTHONPATH:~/dev/personal-photos-model/lib
* Make sure Google Protobufs is installed: sudo pip -g install protobuf


Current Experiment:
* Trying to train Caffe model on LFW (Labeled Faces in the Wild) data set, where input are two LFW images and output is a binary match/not-match classifier. Once I have initial "hello world" model I will then experiment with its hyperparameters. The theory is that I can then use this trained model to do "transfer learning" over to an individuals personal photo set and small collection of labels. For simplicity I am starting with
grayscale photos and a siamese network based on the MNist convolutional network.

To generate data for the current experiment (only has to be done once):
./src/siamese_network_bw/prepare_data.py

To run the current experiment:

./src/siamese_network_bw/train_siamese.sh "_lr_0_001" "(lr = 0.001)"

This would run the Caffe solver and generate plotted test/validation results in the file
src/siamese_network_bw/graphs/siamese_network_bw_lr_0_001. It would also include the text
"(lr = 0.001)" on this graph to help with hyperparameter tuning.

Prior Experiments and Results:
