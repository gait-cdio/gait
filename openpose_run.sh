#!/bin/sh
echo "Running OpenPose on $1 ..."
mkdir -p ./openpose-data/$1
cd $HOME/openpose
./build/examples/openpose/openpose.bin -video $HOME/PycharmProjects/trackingtest/$1 -write_keypoint_json $HOME/PycharmProjects/trackingtest/openpose-data/$1/ -no_display
cd $HOME/PycharmProjects/trackingtest/
echo "Finished running OpenPose."

