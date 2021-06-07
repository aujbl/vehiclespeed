import cv2
import matplotlib.pyplot as plt

plt.figure()
img1 = cv2.imread('cap1.jpg')
img2 = cv2.imread('cap2.jpg')

raw = img1
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (7, 7), 0)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray2 = cv2.GaussianBlur(gray2, (7, 7), 0)

gray = cv2.absdiff(gray1, gray2)

plt.imshow(gray, cmap='gray')
plt.show()
