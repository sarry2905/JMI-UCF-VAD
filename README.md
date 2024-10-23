
# JMI-UCF-VAD

### This repo contains the implementation of paper - *"Video Anomaly Detection using Temporal Self Attention"* on JMI-UCF Campus Dataset.

---

* Extract the frames from the input video.  


* Use the I3D model to extract the features from the frames.<ins>[pytorch-I3D](https://github.com/piergiaj/pytorch-i3d)</ins> can be referred for details.


* Create *'train.list'* and *'test.list'* files to contain the list of the full paths to each and every extracted features.


* Adjust hyperparameter values in *'option.py'*


* Using *'main.py'* train and test the designed model on the dataset.

---
