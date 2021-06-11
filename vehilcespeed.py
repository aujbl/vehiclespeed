import cv2
import cv2.cv2
import numpy as np
import math
import matplotlib.pyplot as plt

from Detector import motion_detector, bounds_filter

# 视频帧率: 30帧/s

def coils_pixels(frame, rect):
    # 计算虚拟线圈内部的平均像素变化量
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    rows, cols = gray.shape
    (cen_x, cen_y), (r_w, r_h), angle = rect
    M = cv2.getRotationMatrix2D(rect[0], angle, 1)
    gray = cv2.warpAffine(gray, M, (cols, rows))
    cut = gray[int(cen_y-r_h/2):int(cen_y+r_h/2), int(cen_x-r_w/2):int(cen_x+r_w/2)]
    pixels = sum(sum(cut))
    mean_pixels = pixels / (r_h*r_w)
    return mean_pixels

def show(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame, cmap='gray')
    plt.pause(0.001)
    # plt.show()
    plt.clf()

def coils_boxs(frame, box, frame_size=None):
    # 在图像中绘制虚拟线圈
    if frame_size:
        frame = cv2.resize(frame, frame_size)
    frame = cv2.drawContours(frame, [box[0]], 0, (0, 0, 255), 2)
    frame = cv2.drawContours(frame, [box[1]], 0, (0, 255, 0), 2)
    return frame


def main():
    cap = cv2.VideoCapture('video3.mp4')
    # 设置两个虚拟线圈的位置， 用矩形框在图像中给出
    rect1 = ((290, 340), (120, 5), -3)
    rect2 = ((360, 370), (200, 5), -4)
    box1, box2 = cv2.boxPoints(rect1), cv2.boxPoints(rect2)
    box = (np.int0(box1), np.int0(box2))
    plt.figure(0, figsize=(16, 8))
    # 速度，线圈距离，时间
    velocity, len_coils, time = 0, 10, 0

    ret, frame = cap.read()
    # 在图像帧中画出线圈位置
    frame = coils_boxs(frame, box)
    frames = [frame] * 2

    thresh = motion_detector(frames, enhencer=cv2.MORPH_ELLIPSE)
    contours, _ = cv2.findContours(motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounds = [cv2.boundingRect(contour) for contour in contours]
    bounds = bounds_filter(bounds)

    # 计算线圈位置是否有运动物体
    # mean_coil_1 = coils_pixels(frame, rect1)
    # mean_coil_2 = coils_pixels(frame, rect2)
    # 标志位，记录是否有汽车进入
    car_enter = False
    while ret:
        frame = coils_boxs(frame, box)
        coil_1 = coils_pixels(frame, rect1)
        coil_2 = coils_pixels(frame, rect2)
        diff_1 = abs(mean_coil_1 - coil_1)
        diff_2 = abs(mean_coil_2 - coil_2)
        frame = cv2.putText(frame, str('%.2f' % diff_1), rect1[0], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)
        frame = cv2.putText(frame, str('%.2f' % diff_2), rect2[0], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)
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
        show(frame)
        ret, frame = cap.read()
    cap.release()


if __name__ == '__main__':
    main()
