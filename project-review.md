# Project Review -- Traffic Sign Recognition

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This self-review is a follow up on the [project writeup](https://github.com/jingz8804/CarND-Traffic-Sign-Classifier-Project/blob/master/Project%20Writeup.md), mostly focusing on things I've tried based on the ideas received from the code review. 

As mentioned in the project [README](https://github.com/jingz8804/CarND-Traffic-Sign-Classifier-Project), we use convolutional neural networks to classify traffic signs. The data involved is the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, we also tried out the model on images of German traffic signs found on the web.

Using the LeNet-5 network architecture with dropout, the model I trained achieved near 100% training accuracy, 96.5% validation accuracy and 93.7% test accuracy. I then made some modification to the network according to the paper [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). The performance did not improve much but I'm not trying to replicate the exact thing either. 

Suggestions from the code review
---
There are a few suggestions from the reviewer:

* Based on the imbalanced data distribution across the 43 classes:
  * Ensuring the distribution of samples across different classes is well balanced.
  * Augmenting the training data with image processing techniques (shifts, brightness adjustments, etc.) to create more diverse examples for the model to learn from.

* General network architecture and optimization suggestions:
  * Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift - 2015 paper
  * A newer optimizer gaining popularity is the Nesterov Adam (Nadam) optimizer.
  * To improve the model's performance, you can also experiment with increasing the number of convolution or fully connected layers in your network (in addition to the suggestions provided above).

Incorporating the ideas
---
There are certainly rooms for improvements but we don't want to jump right in. We should try to do it logically as opposed to do it in an adhoc way. 

We can see that there isn't much avoidable bias (the gap between training and human accuracy) left but there is a 3.5% gap between training and validation accuracy. We can safely attribute it to the variance since the distributions across the training, validation and test sets are very similar. 

In order to reduce gaps caused by variance, things we can try are:
* applying regularization like L2 or dropout
* getting more data
* search for a better network architecture and/or the hyperparameters

Here I'm using the modified LeNet network. The first thing I tried was to reduce the keep prob in the dropout layer to 0.35 (which increases the probability of a node to be shut down during training).