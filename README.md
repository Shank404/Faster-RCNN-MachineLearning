# Faster-RCNN---German_speedsign_identification

### This was our first artificial intelligence project :)
Our tool of choice was MATLab so all code is written in MATLab language.

Our goal was to design a neural network which can identify different german speedsigns.
First we build up a data collection where we collected pictures of different signs.
Furthermore we classified the data in "30 km/h","50 km/h","60 km/h" and "No speedsigns/other signs".
We constructed a deep network in different variants.
First we used a faster RCNN (ResNet50) for region detection linked with a simple CNN for classifying data.
Maximizing acurracy was our goal while we optimized parameters.

Second try was a setup with a faster RCNN (ResNet50) for region detection linked with a AlexNet for classifying data.
The AlexNet is a neural network which performs very good in recognizing objects.
We used a technique called transfer learning in which we relearned the last three layers.
With this we optimized accuracy of our deep learning network significantly.

## 1. Faster RCNN (ResNet50) + simple CNN
#### ResNet50 => 73% accuracy
#### CNN =>      90.87% accuracy

## 2. Faster RCNN (ResNet50) + AlexNet
#### ResNet50 => 73% accuracy
#### AlexNet =>  98.87% accuracy
