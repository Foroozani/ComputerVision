# Computer Vision (CV)

[![TensorFlow 2.2](https://img.shields.io/badge/TensorFlow-2.2-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0) [![Documentation Status](https://readthedocs.org/projects/tensorflow-object-detection-api-tutorial/badge/?version=latest)](http://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/?badge=latest)
[![Pillow](https://readthedocs.org/projects/pillow/badge/?version=latest)](https://pillow.readthedocs.io/?badge=latest)
[![Python 3.6+](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![PyPI Version](https://img.shields.io/pypi/v/imageio.svg)](https://pypi.python.org/pypi/imageio/)
[![Documentation Status](https://readthedocs.org/projects/imageio/badge/?version=latest)](https://imageio.readthedocs.io)
[![PyPi Download stats](http://pepy.tech/badge/imageio)](http://pepy.tech/project/imageio)
[![](https://img.shields.io/badge/torchvision-v0.1.6-green)](https://pypi.org/project/torchvision/0.1.6/)

---
Let us start with simple definition: **Predicting the location of the object along with the class is called object Detection**. In place of predicting the class of object from an image, we now have to predict the class as well as a rectangle(called bounding box) containing that object. It takes 4 variables to uniquely identify a rectangle. Object Detection is modeled as a classification problem where we take windows of fixed sizes from input image at all the possible locations feed these patches to an image classifier. Each window is fed to the classifier which predicts the class of the object in the window( or background if none is present). There are various methods for object detection like RCNN, Faster-RCNN, [SSD](https://github.com/Foroozani/ComputerVision/tree/main/object_detection_SSD), [YOLO](https://github.com/Foroozani/Object_detect_YOLOV3V4) etc. 


<p align="center">
  <img width="460" height="300" src="https://github.com/Foroozani/ComputerVision/blob/main/image/ssd-yolo.png">
</p>


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
## Single Shot Detector(SSD):

**S**ingle **S**hot **D**etector achieves a good balance between speed and accuracy. SSD runs a convolutional network on input image only once and calculates a feature map. Now, we run a small 3×3 sized convolutional kernel on this feature map to predict the bounding boxes and classification probability. SSD also uses anchor boxes at various aspect ratio similar to Faster-RCNN and learns the off-set rather than learning the box. In order to handle the scale, SSD predicts bounding boxes after multiple convolutional layers. Since each convolutional layer operates at a different scale, it is able to detect objects of various scales.

That’s a lot of algorithms. Which one should you use? Currently, Faster-RCNN is the choice if you are fanatic about the accuracy numbers. However, if you are strapped for computation(probably running it on Nvidia Jetsons), SSD is a better recommendation. Finally, if accuracy is not too much of a concern but you want to go super fast, YOLO will be the way to go. First of all a visual understanding of speed vs accuracy trade-off:

![](https://github.com/Foroozani/ComputerVision/blob/main/image/comparision.png)


Install the library [imageio](https://imageio.readthedocs.io/en/stable/userapi.html)
```bash 
pip install imageio
pip install imageio-ffmpeg
2.9.0
```

[VOC Dataset ](http://host.robots.ox.ac.uk/pascal/VOC/index.html)






## Module 3: Image Creation with GANs
















---
**Refrences**

https://cv-tricks.com/object-detection/faster-r-cnn-yolo-ssd/
