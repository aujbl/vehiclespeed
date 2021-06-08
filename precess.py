import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.figure(0, figsize=(16, 8))

def preprocess(frame, k_size=5, fileter='Gauss', threshold=60, op=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (k_size, k_size), 0.) if fileter == 'Gauss' \
            else cv2.medianBlur(gray, k_size)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    if op:
        op = cv2.getStructuringElement(op, (k_size, k_size))
        thresh = cv2.erode(thresh, op)
        thresh = cv2.dilate(thresh, op)
    return gray, thresh

def frame_diff(frames):
    diff_1 = cv2.absdiff(frames[1], frames[0])
    diff_2 = cv2.absdiff(frames[2], frames[1])
    diff = cv2.absdiff(diff_2, diff_1)
    return diff

cap = cv2.VideoCapture('video2.mp4')
ret, frame = cap.read()
_, frame = preprocess(frame, k_size=5, fileter='m', op=cv2.MORPH_ELLIPSE)
frames = [frame] * 3
while ret:
    ret, frame = cap.read()
    _, frame = preprocess(frame, k_size=5, fileter='m', op=cv2.MORPH_ELLIPSE)
    frames.pop(0)
    frames.append(frame)
    diff = frame_diff(frames)

    contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        box = cv2.boundingRect(contour)
        x, y, w, h = box
        diff = cv2.rectangle(diff, (x, y), (w, h), (0, 255, 0))

    plt.imshow(diff, cmap='gray')
    plt.pause(0.0001)
    plt.clf()
cap.release()

# plt.show()
