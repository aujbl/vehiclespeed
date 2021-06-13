import cv2
import cv2.cv2
import numpy as np
import math
import matplotlib.pyplot as plt

from Detector import motion_detector, bounds_filter

# 视频帧率: 30帧/s


def count_coils_pixels(gray, rect):
    # 计算虚拟线圈内部的平均像素变化量
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    rows, cols = gray.shape
    (cen_x, cen_y), (r_w, r_h), angle = rect
    M = cv2.getRotationMatrix2D(rect[0], angle, 1)
    gray = cv2.warpAffine(gray, M, (cols, rows))
    cut = gray[int(cen_y-r_h/2):int(cen_y+r_h/2), int(cen_x-r_w/2):int(cen_x+r_w/2)]
    pixels = sum(sum(cut))
    mean_pixels = pixels / (r_h*r_w)
    return mean_pixels


def show(orig_img, motion, wait_time=100):
    cv2.imshow('orig_img', orig_img)
    cv2.imshow('motion', motion)
    cv2.waitKey(wait_time)


def draw_coils_boxes(frame, box, frame_size=None):
    # 在图像中绘制虚拟线圈
    if frame_size:
        frame = cv2.resize(frame, frame_size)
    frame = cv2.drawContours(frame, [box[0]], 0, (0, 0, 255), 2)
    frame = cv2.drawContours(frame, [box[1]], 0, (0, 255, 0), 2)
    return frame


def update_frames(orig_img, frames):
    frame = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    frames.pop(0)
    frames.append(frame)
    return frames


def main():
    cap = cv2.VideoCapture('video3.mp4')
    # 设置两个虚拟线圈的位置， 用矩形框在图像中给出
    rect1 = ((290, 340), (120, 5), -3)
    rect2 = ((390, 365), (150, 5), -4)
    box1, box2 = cv2.boxPoints(rect1), cv2.boxPoints(rect2)
    box = (np.int0(box1), np.int0(box2))
    plt.figure(0, figsize=(16, 8))
    # 速度km/h，线圈间距：4m，时间
    velocity, coils_intrval, time = 0, 3.5, 0

    ret, orig_img = cap.read()
    frame = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    frames = [frame] * 2

    # 标志位，记录是否有汽车进入
    car_enter, cnt_frames = False, 1
    while ret:
        frames = update_frames(orig_img, frames)
        motion = motion_detector(frames)
        cnt1 = count_coils_pixels(motion, rect1)
        cnt2 = count_coils_pixels(motion, rect2)
        cv2.putText(orig_img, str('%.2f' % cnt1), (rect1[0][0]-5, rect1[0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(orig_img, str('%.2f' % cnt2), (rect2[0][0]-5, rect2[0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        if not car_enter:
            cv2.putText(orig_img, 'total jump frames: %d ' % (cnt_frames), \
                        (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            time = cnt_frames / 30
            cv2.putText(orig_img, 'velocity: %.2f km/h' % (coils_intrval / time * 3.6), \
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            if cnt1 > 20 and cnt2 == 0:
                car_enter = True
                cnt_frames = 1
        else:
            if cnt2 <= 20:
                cnt_frames += 1
            else:
                car_enter = False
            cv2.putText(orig_img, 'car coming, jump frames: %d ' % (cnt_frames), \
                    (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        orig_img = draw_coils_boxes(orig_img, box)
        contours, _ = cv2.findContours(motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounds = [cv2.boundingRect(contour) for contour in contours]
        bounds = bounds_filter(bounds)
        for bound in bounds[:3]:
            x, y, w, h = bound
            cv2.rectangle(orig_img, (x, y), (x + w, y + h), (0, 255, 0))
        show(orig_img, motion, 30)
        ret, orig_img = cap.read()



    cap.release()


if __name__ == '__main__':
    main()
