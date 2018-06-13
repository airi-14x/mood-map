from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
#from face_cnn import FaceCNN
from cnn_model_v01 import cnn
from globalcontrast import GCNorm
trans = transforms.Compose([transforms.ToPILImage(), transforms.Resize((48, 48)),
                            transforms.Grayscale(), GCNorm()])
#face_cnn = FaceCNN()
face_cnn = cnn()
#face_cnn.load_state_dict(torch.load('best_binary.pth'))
face_cnn.load_state_dict(torch.load('cnn_model_v01_trainingWeights.pt'))
face_cnn.eval()

conf = .5  # hard coded confidence in a face detection.

# load deep net. 
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "model.caffemodel")

vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame resize to have max width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
 
    # convert to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # get the detections and predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # get assoc pred confidence
        confidence = detections[0, 0, i, 2]
        if (confidence < conf):
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x, y, w, h) = box.astype("int")
        sub_img = frame[y:y+h, x:x+w]
        sub_img = trans(sub_img)
        out = face_cnn(sub_img.view(1,1,48,48))
        _, pred = torch.max(out.data, 1)

        # text gen: hardcoding labels. :/
        if (pred == 0):
            text = "Angry"
        elif (pred == 1):
            text = "Afraid"
        elif (pred == 2):
            text = "Happy"
        elif (pred == 3):
            text = "Sad"
        elif (pred == 4):
            text = "Surprised"
        else:
            text = "Neutral"

	# draw the bounding box
        ty = y - 10 if y - 10 > 10 else y + 10
        cv2.rectangle(frame, (x, y), (w, h),(0, 255, 255), 2)
        cv2.putText(frame, text, (x, ty),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.45, (0, 255, 255), 1)

    cv2.imshow("Frame", frame)
	
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
vs.stop()
