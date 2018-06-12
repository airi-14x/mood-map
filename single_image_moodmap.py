import cv2
import sys
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from face_cnn import FaceCNN

trans = transforms.Compose([transforms.ToPILImage(), transforms.Resize((48, 48)),
                            transforms.Grayscale(), transforms.ToTensor(),
                            transforms.Normalize([0],[1])])

face_cnn = FaceCNN()
face_cnn.load_state_dict(torch.load('best_binary.pth'))
face_cnn.eval()

imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#detection
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

#print("Found {0} faces!".format(len(faces)))
# rectangle time :) 
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    sub_face = image[y:y+h, x:x+w]
    sub_img = trans(sub_face)
    out = face_cnn(sub_img.view(1,1,48,48))
    _, pred = torch.max(out.data, 1)

    #print(pred)

    # this is the simplified version -- works for only 2 emotions right now. 
    if (pred == 0):
        cv2.putText(image, "Disgust", (x, y-5),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.45, (0,255,0),1)
    else:
        cv2.putText(image, "Not Disgust", (x, y-5),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.45, (0,255,0),1)
    

cv2.imshow("observed emotions", image)
cv2.waitKey(0)
