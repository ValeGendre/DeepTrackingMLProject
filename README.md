# DeepTrackingMLProject

Problem Statement:
Our aim is to track mice in a Smart Vivarium in an unsupervised/self-supervised manner. We are working with a newly collected dataset collected by Dr. Benoit Labonté’s team at the CERVO Brain Research Centre, that is affiliated with the Smart Biomedical Microsystems Laboratory at ULaval directed by Benoit Gosselin. The aim of Dr. Labonté’s team is to understand complex behaviour of mice in a Smart Vivarium; the first step of which is to track the mice’s movement. A common way to do so is using RFID chips. RFID chips can give us an estimate of the location of the mice. However, the data collected using RFID can be inaccurate because the Vivarium is separated into grids and the RFID only detects if the mouse is in that grid. Therefore, the team decided to use Computer Vision and Deep Learning to track the mice. However, labelling the locations of the mice in the video can be a laborious task, therefore we want to help them achieve this first step while trying to maintain a good accuracy.
The problem we are addressing is tracking in videos without labelled data. We use the algorithm proposed in [2] wherein the framework is based on a Siamese correlation filter network. The fundamental idea behind the algorithm is that the tracker should be effective in both the forward and backward predictions.

Dataset used:
Their team has recorded many hours of Full HD vivarium video. Multiple mice are living in, and the vivarium is composed of several areas. There are two types of videos being collected:
1. B&W night vision of the vivarium and the grey mice are clearly distinguishable from the light background (1Go). More videos of this type are being recorded and can be used for this project.
2. Others are color videos, where the mice are less distinguishable and a grid is obstructing the view (+200Go, ~25h).

Method:
The dataset we’re using is raw which means that there are no labels whatsoever. Most state-of-the-art tracking algorithms rely on labelled data for training, therefore we’re using a self-supervised deep neural network. Basically a self-supervised network creates its own supervision during the training. A robust tracking model achieves the same results if you’re tracking forward or backward in time. Therefore, in [2], the forward tracking is used as labels when tracking the same video backward in time. Then we compute the mean squared error between forward and backward tracking. This loss is then back propagated following the Siamese correlation filter methods.

Possible improvements
1. If we do not achieve good tracking using the current model, we will perform camera calibration to undistort the image.
2. Compare accuracies on B&W night vision dataset with coloured dataset and grid-obstructed dataset.
3. Compare our results with results achieved on this dataset using code of [1].

References:
[1] Joint-task Self-supervised Learning for Temporal Correspondence [Li et. al]
[2] Unsupervised Deep Tracking [Wang et. al]
[3] DCFNET: Discriminant Correlation Filters Network For Visual Tracking [Wang et. al]
