# Computer Vision (CV)

[![TensorFlow 2.2](https://img.shields.io/badge/TensorFlow-2.2-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0) [![Documentation Status](https://readthedocs.org/projects/tensorflow-object-detection-api-tutorial/badge/?version=latest)](http://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/?badge=latest)
[![Pillow](https://readthedocs.org/projects/pillow/badge/?version=latest)](https://pillow.readthedocs.io/?badge=latest)
[![Python 3.6+](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

---
Let us start with simple definition: **Predicting the location of the object along with the class is called object Detection**. In place of predicting the class of object from an image, we now have to predict the class as well as a rectangle(called bounding box) containing that object. It takes 4 variables to uniquely identify a rectangle. Object Detection is modeled as a classification problem where we take windows of fixed sizes from input image at all the possible locations feed these patches to an image classifier. Each window is fed to the classifier which predicts the class of the object in the window( or background if none is present). There are various methods for object detection like RCNN, Faster-RCNN, SSD, [YOLO](https://github.com/Foroozani/Object_detect_YOLOV3V4) etc. 


## Module 1: Face detection with OpenCV
### (a) - The Viola-Jones algorithm 
This is one of the most powerful to date algorithms for computer vision developed by [P. Viola and M. Joens](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.10.6807&rep=rep1&type=pdf). This algorithm lies at the foundation of [OpenCV](https://github.com/opencv/opencv) library. I downgrade opencv. check the version

```bash
pkg-config --modversion opencv

3.2.0
```
in case if it is not found, try `sudo apt-get install libopencv-devsudo`. For this virtual-environment I have `python3.6`. All the libraries and dependencies verion can be find in *environment.yml*. Main libraries installed with:

```bash 
$ pip install torchvision==0.1.6
$ pip3 install torch==0.3.1
$ conda install -c menpo opencv3

```

### (b) - Emotion detection
Many companies today use CV in their core business to detect emotions. For example, Apple bought Emotient, a startup that builds CV tools to recognize people's feelings.
Building an AI that sees human emotions can be highly valuable in some markets, like recomender system or self-driving car. Here is an example to detect one motion: Happiness :) 





Additional reading:

- Paul Viola & Michael Jones, 2001 [Rapid bject Detection using a Boosted Cascade of Simple Features](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.10.6807&rep=rep1&type=pdf)

- Kinh Tieu & Paul Viola, 2000 [Boosting Image Retrieval](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.136.2419&rep=rep1&type=pdf)    


## Module 2: Object Detection With SSD


## Module 3: Image Creation with GANs
