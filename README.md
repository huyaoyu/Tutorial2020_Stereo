# Tutorial2020_Stereo: The Exercises on Recent Deep-Learning Stereo Vision Models. Prepared for the Summer School 2020 at the [airlab][airlabss] in Carnegie Mellon University

[airlabss]: https://theairlab.org/summer2020

This repository contains the exercise code for the course of __Recent Advances of Binocular Stereo Vision__ held by the [airlab][airlabss] in Carnegie Mellon University as a part of the Summer School 2020. 

The course covers both recent non-learning and learning based methods. This repository contains the learning-based models discussed in the lecture. The non-learning part cab be found [here](http://github.com).

# Models

This exersice code presents two popular deep-learning structures for passive binocular stereo vision, namely, the 3D cost volume structure [ref] and the cross-correlation structure [ref].

## 3D cost volume

The backbone comes from the PSMNet. The origina PSMNet is modified such that it also estimates the uncertainty of its own disparity prediction. This modified model is called PSMNU. Please refer to [ref PSMNNU] for more detail about PSMNU. 

## Cross-correlation

The implementaion of the cross-correlation is mainly from the PWC-Net which does optical flow estimation. The model provided here is modified to match the current PyTorch version 1.5. Cross-correlation is only performed along the x-axis since we are only care about the disparity not the optical flow.

