import time

import cv2
import numpy as np
from scipy.io import loadmat
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from utils import normalise, flatten_3d, cropbox
from nms import nms
from svm import SVM

PATH = "D:/OneDrive/Documents/GitHub/AdvancedML/practical-face-detection/images/"

# Load data
pos = loadmat(PATH + 'possamples.mat')["possamples"]
neg = loadmat(PATH + 'negsamples.mat')["negsamples"]

# Show average image
fig, ax = plt.subplots(1, 2)
ax[0].imshow(Image.fromarray(np.mean(pos, axis=2)))
ax[1].imshow(Image.fromarray(np.mean(neg, axis=2)))
# plt.show()

# Normalize
pos = np.nan_to_num(normalise(pos))
neg = np.nan_to_num(normalise(neg))

# Flatten 3d to 2d
pos = flatten_3d(pos)
neg = flatten_3d(neg)

# Label vector
pos_labels = np.ones((pos.shape[0], 1))
neg_labels = -np.ones((neg.shape[0], 1))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(np.concatenate((pos, neg), axis=0),
                                                    np.concatenate((pos_labels, neg_labels), axis=0),
                                                    test_size=0.33,
                                                    random_state=33)

del pos, neg, pos_labels, neg_labels

# 1. Train and test SVM classifier
clf = SVM(C=0.001)
start = time.time()
print(f"Training phase with {X_train.shape[0]} samples")
clf.fit(X_train, y_train)
print(f"Done in {time.time() - start}")

print(f"Evaluation phase with {X_test.shape[0]} samples")
y_pred = clf.predict(X_test)
print(f"Accuracy : {sum(y_pred == y_test)/len(y_test):.3%}%")

# 2. Find best model
lst_c = [1000, 100, 10, 1, .1, .01, .001, .0001, .00001]
best = -np.inf
for c in lst_c:
    model = SVM(C=c, kernel='linear')
    model.fit(X_train, np.ravel(y_train))
    acc = sum(model.predict(X_test) == np.ravel(y_test)) / y_test.shape[0]
    if best < acc:
        best = acc
        cbest = c
        bestmodel = model
    print(f"c = {c} --- accuracy = {acc}")

# 3. Face detection
img4 = np.array(Image.open(PATH + r'/img4.jpg'))
img4 = img4.mean(axis=2, keepdims=True)

# List of bboxes
h, w, c = img4.shape
xx, yy = np.meshgrid(range(0, w-24), range(0, h-24))
bbox = np.concatenate((xx.reshape(-1, 1),
                       yy.reshape(-1, 1),
                       xx.reshape(-1, 1)+24-1,
                       yy.reshape(-1, 1)+24-1),
                      axis=1)

# Get array given bbox
imgcropall = np.zeros((24, 24, len(xx.ravel())))
for i in range(len(bbox)):
    imgcropall[:, :, i] = np.squeeze(cropbox(img4, bbox[i, :]))
imgcropall = flatten_3d(normalise(imgcropall))

# Predict using best model
conf = clf.project(imgcropall)
conf = np.nan_to_num(conf)

# Get top20 bboxes from predicted array
top_bbox = bbox[np.argsort(conf.ravel())[::-1][:20]]

img_top20 = img4.copy()
for i in top_bbox:
    img_top20 = cv2.rectangle(img_top20, (i[0], i[2]), (i[1], i[3]), (255, 0, 0))

# Get bboxes using NMS algo from top20 bboxes
img_nms = img4.copy()
for i in nms(top_bbox, 0.7):
    img_nms = cv2.rectangle(img_nms, (i[0], i[2]), (i[1], i[3]), (255, 0, 0), 1)

# Plot top20 vs nms
fig, ax = plt.subplots(1, 2)
ax[0].imshow(img_top20)
ax[0].title.set_text("Draw top 20 bboxes")
ax[1].imshow(img_nms)
ax[1].title.set_text("NMS")
plt.show()
