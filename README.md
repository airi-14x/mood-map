# mood-map
Project for AI4SocialGood Lab 2018

**By: Anne Meisner, Airi Chow, Anna Ilina**

We are building an emotion recognition app for people suffering from Alexithymia (difficulty in identifying other's emotions). We are training a CNN model using pytorch to classify the seven basic emotions (0=Angry , 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). Training data is taken from a kaggle dataset [1], with ~28k images for training. Using open CV, we can do facial recognition from live video (webcam, etc) and pass the cropped face images into our CNN model to classify emotions in real time. We plan to integrate this into an Android application.

To install required packages, run:
```python
pip install -r requirements.txt
```

[1] https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge 
