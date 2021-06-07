import cv2
import matplotlib.pyplot as plt

fig = plt.figure()
img1 = cv2.imread('cap1.jpg')
img1 = cv2.GaussianBlur(img1, (5, 5), 0)
img2 = cv2.imread('cap2.jpg')
img2 = cv2.GaussianBlur(img2, (5, 5), 0)

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
diff = img1 - img2
# diff = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)

plt.imshow(diff, cmap='gray')
plt.show()

diff[diff < 150] = 0
# diff = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)
plt.imshow(diff, cmap='gray')
plt.show()
