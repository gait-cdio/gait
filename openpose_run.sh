#!/bin/sh
echo "I AM DOING THE THING MR"
mkdir ./openpose-data/$1
cd $HOME/openpose
./build/examples/openpose/openpose.bin -video $HOME/PycharmProjects/trackingtest/input-videos/$1 -write_keypoint_json $HOME/PycharmProjects/trackingtest/openpose-data/$1/
cd $HOME/PycharmProjects/trackingtest/
echo "OOOWEEE LOOK AT ME I'M MR MEESEEKS"

