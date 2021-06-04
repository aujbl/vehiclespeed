import cv2
import cv2.cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# 视频帧率: 30帧/s
cap = cv2.VideoCapture('uusgu-1bxuj.mp4')
rect1 = ((342, 702), (7, 40), 55)
rect2 = ((252, 618), (7, 40), 55)
box1, box2 = cv2.boxPoints(rect1), cv2.boxPoints(rect2)
box1, box2 = np.int0(box1), np.int0(box2)
plt.figure(0, figsize=(16, 8))

def coils_pixels(frame, rect):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rows, cols = gray.shape
    (cen_x, cen_y), (r_h, r_w), angle = rect
    M = cv2.getRotationMatrix2D(rect[0], -angle, 1)
    gray = cv2.warpAffine(gray, M, (cols, rows))
    cut = gray[int(cen_y-r_h/2):int(cen_y+r_h/2), int(cen_x-r_w/2):int(cen_x+r_w/2)]
    pixels = sum(sum(cut))
    mean_pixels = pixels / (r_h*r_w)
    return mean_pixels

def show(frame):
    frame = cv2.drawContours(frame, [box1], 0, (0, 0, 255), 2)
    frame = cv2.drawContours(frame, [box2], 0, (0, 255, 0), 2)
    plt.imshow(frame)
    plt.pause(0.001)
    plt.clf()

# def put_text():


def main():
    ret, frame = cap.read()
    frame_size = (540, 960)
    velocity, len_coils, time = 0, 10, 0
    frame = cv2.resize(frame, frame_size)
    mean_coil_1 = coils_pixels(frame, rect1)
    mean_coil_2 = coils_pixels(frame, rect2)
    car_enter = False
    while ret:
        frame = cv2.resize(frame, frame_size)
        coil_1 = coils_pixels(frame, rect1)
        coil_2 = coils_pixels(frame, rect2)
        diff_1 = abs(mean_coil_1 - coil_1)
        diff_2 = abs(mean_coil_2 - coil_2)
        # print('diff_1:%.2f, diff_2:%.2f' % (diff_1, diff_2))
        if car_enter:
            cnt += 1
            frame = cv2.putText(frame, "car comming", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        else:
            frame = cv2.putText(frame, "pass time " + str('%.2f s' % time), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            frame = cv2.putText(frame, "The speed is " + str('%.2f' % velocity) + "Km/h", (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        if diff_1 > 5 and car_enter == False:
            car_enter = True
            cnt = 1
        if diff_2 > 5 and car_enter:
            car_enter = False
            # 30帧/s, time/s, len_coils = 10m
            time = 1 / 30 * cnt
            velocity = len_coils / time * 3.6  # km/h
            # print('cnt: ', cnt)
            # print('time:%.2f s' % time)
            # print('velocity:%.2f km/h' % velocity)
        show(frame)
        ret, frame = cap.read()

if __name__ == '__main__':
    main()