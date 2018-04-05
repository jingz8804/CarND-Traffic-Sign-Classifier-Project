# Project Review -- Traffic Sign Recognition

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This self-review is a follow up on the [project writeup](https://github.com/jingz8804/CarND-Traffic-Sign-Classifier-Project/blob/master/Project%20Writeup.md), mostly focusing on things I've tried based on the ideas received from the code review as well as things I learned from other places on tuning neural networks. 

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

There isn't much avoidable bias (the gap between training and human accuracy) left but there is a 3.5% gap between training and validation accuracy. We can safely attribute it to the variance since the distributions across the training, validation and test sets are very similar. 

In order to reduce gaps caused by variance, things we can try are:
* applying regularization like L2 or dropout
* getting more data
* search for a better network architecture and/or the hyperparameters

Here I'm using the modified LeNet network. The first thing I tried was to reduce the keep prob in the dropout layer to 0.35 (which increases the probability of a node to be shut down during training). 
* This ends with training accuracy 99.4% and validation accuracy 96.3%. 
* Since there are oscilations in the last a few epochs, I reduced the learning rate (by about 50%) and continued the training for 30 extra epochs, and repeated once. The learning rate changes from 0.001 to 0.0005 and eventually 0.0002. This gave me 99.8% training accuracy and 97.2% validation accuracy.
* Keep doing this didn't help anymore so I stopped there. We have bigger fish to fry even though there might be improvement down this path.

What about generating more data? The data is imbalanced. As suggested, I created some augmented data. 
* Now we have 5000 examples per class. With just 60 Epochs and 0.001 learning rate, we got 97.9% training accuracy and 97.3% validation accuracy. 
* Reduced the learning rate to 0.0005 and continued with 30 epochs. The validation accuracy stays 97.5% while training accuracy reached around 98.7%.
* I tried with 10000 examples per class and repeated the process above. It stayed around 98.3% training accuracy and 97.6% validation accuracy. 

Take aways
---
1. Don't jump right in to tune the parameters. Analyze the situation first. Which one is worth working on? The bias or the variance?
2. Data matters (more). With more data, you can train fewer epochs to achieve the same performance.

Future work
---
The followings are just things I don't have to do now as there are other projects down the road:
1. Try a different network architecture. There are many other superior networks on github and in the forum, although I'm more interested to know how one can come up with a good architecture for this type of problem. 
2. Get a GPU. It took too long to train the model on my laptop, although it did gave me the opportunity to do other things during that time rather than to sit and watch the screen. 
