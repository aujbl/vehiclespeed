import cv2
import cv2.cv2
import numpy as np
import math
import matplotlib.pyplot as plt

from Detector import motion_detector, bounds_filter


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
    frame = cv2.drawContours(frame, [box[2]], 0, (255, 0, 0), 2 )
    return frame


def update_frames(orig_img, frames):
    frame = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    frames.pop(0)
    frames.append(frame)
    return frames


def main():
    cap = cv2.VideoCapture('video3.mp4')
    # 设置两个虚拟线圈的位置， 用矩形框在图像中给出
    rect_1 = ((290, 340), (120, 5), -3)
    rect_2 = ((390, 365), (150, 5), -4)
    rect_stop = ((640, 450), (200, 5), -9)
    box_1, box_2, box_stop = cv2.boxPoints(rect_1), cv2.boxPoints(rect_2), cv2.boxPoints(rect_stop)
    box = (np.int0(box_1), np.int0(box_2), np.int0(box_stop))
    plt.figure(0, figsize=(16, 8))
    # 速度km/h，线圈间距：4m，时间
    velocity, coils_interval, time, stop_lines = 0, 3.5, 0, 2.5

    ret, orig_img = cap.read()
    frame = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    frames = [frame] * 2

    # 标志位，记录是否有汽车进入
    car_enter_flag, crush_flag, cnt_frames, cnt_crush_frames = False, False, 1, 1
    while ret:
        frames = update_frames(orig_img, frames)
        motion = motion_detector(frames)
        cnt_1 = count_coils_pixels(motion, rect_1)
        cnt_2 = count_coils_pixels(motion, rect_2)
        cnt_stop = count_coils_pixels(motion, rect_stop)
        cv2.putText(orig_img, str('%.2f' % cnt_1), (rect_1[0][0]-5, rect_1[0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(orig_img, str('%.2f' % cnt_2), (rect_2[0][0]-5, rect_2[0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(orig_img, str('%.2f' % cnt_stop), (rect_stop[0][0]-5, rect_stop[0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255), 1)
        # 标志位跳变过程
        if not car_enter_flag:
            if cnt_1 > 20 and cnt_2 == 0:
                car_enter_flag = True
                cnt_frames = 1
                crush_flag = False
            if cnt_2 > 20 and cnt_stop < 10:
                cnt_crush_frames += 1
        else:
            if cnt_2 <= 20:
                cnt_frames += 1
            else:
                car_enter_flag = False
                crush_flag = True
                cnt_crush_frames = 1
        # 计算并显示重要信息
        text_1 = 'velocity detection phase: jump frames %d' % cnt_frames
        velocity = (coils_interval/(cnt_frames/30)*3.6)
        text_2 = 'velocity: %.2f km/h' % velocity
        text_3 = 'breast the tape phase: jump frames %d' % cnt_crush_frames
        text_4 = 'predict time: %.2f s, actual time: %.2f s' \
                 % ((3.6*stop_lines/velocity), (cnt_crush_frames/30))
        cv2.putText(orig_img, text_1, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(orig_img, text_2, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(orig_img, text_3, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(orig_img, text_4, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        orig_img = draw_coils_boxes(orig_img, box)
        contours, _ = cv2.findContours(motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounds = [cv2.boundingRect(contour) for contour in contours]
        bounds = bounds_filter(bounds)
        for bound in bounds[:3]:
            x, y, w, h = bound
            cv2.rectangle(orig_img, (x, y), (x + w, y + h), (0, 255, 0))
        show(orig_img, motion, 50)
        ret, orig_img = cap.read()



    cap.release()


if __name__ == '__main__':
    main()
