### Facial Recognition with OpenCV and Python

This is the final project of the course Python Scientifique at Ecole Centrale de Marseille.

Using basic built-in functionalities of OpenCV, namely haar cascades and LBPH classifier, I managed to train a simple model to detect faces and classify them based on the dataset.

### Related works and ressources

This project relies highly on the OpenCV/Python tutorial of freeCodeCamp.org on YouTube. For those who are reading this, I highly recommend that you watch the tutorial at this link: https://www.youtube.com/watch?v=oXlwWbU8l2o. It's also worth mentioning that OpenCV provides a variety of pre-trained haar cascade models for us to use, one of which is used in this particular project: https://github.com/opencv/opencv/tree/4.x/data/haarcascades/haarcascade_frontalface_default.xml. When trying to use this file, do a raw copy and save it in a local file.

The dataset is taken on kaggle: https://www.kaggle.com/apollo2506/facial-recognition-dataset.

### Results and observations

This project has reached a precision (which is based on a simple determination of the correctness of predictions) of 52.37%, which remains very low compared to benchmarks. Considering the little manual work needed to implement a simple computer vision system, I would say that this is an appealing material to have on data-science-related or artificial-intelligence-related Python classes.

Although haar cascades proves to be a simple and effective method to detect faces (as well as other objects), this method hasn't shown great results in my experiment. I would suggest that future work which intends to improve the performance of the system should try to build a CNN (convolutional neural network) using PyTorch or Tensorflow, or any other DL framework.

### Remarks

According to the restrictions on number and size of uploading files on Github, I haven't included the .xml file (haar cascade), the .yaml file (the trained model) and the pictures. If you are interested, I strongly encourage you to look up to the tutorial and feel free to retrain.
