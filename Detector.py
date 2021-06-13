import cv2
import numpy as np
import matplotlib.pyplot as plt


def bounds_filter(bounds, min_area=50):
    # 去掉面积较小的区域
    bounds = [bound for bound in bounds if bound[2]*bound[3] > min_area]
    # 根据矩形框面积进行排序
    bounds.sort(key=lambda bound: bound[2]*bound[3], reverse=True)
    i = 0
    while i < len(bounds):
        x1, y1, w1, h1 = bounds[i]
        new_bounds = []
        for bound in bounds[i+1:]:
            x2, y2, w2, h2 = bound
            if not (x1 <= x2 <= x1+w1 and y1 <= y2 <= y1+h1):
                new_bounds.append(bound)
        i += 1
        bounds = bounds[:i]+new_bounds
    return bounds


def motion_detector(frames, blur_size=5, threshold=15, enhencer=None, e_size=7, iterations=5):
    #  input: 两帧灰度图
    # output： 运动物体的二值图
    diff = cv2.absdiff(frames[0], frames[1])
    diff = cv2.medianBlur(diff, blur_size)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    if enhencer:
        enhencer = cv2.getStructuringElement(enhencer, (e_size, e_size))
        thresh = cv2.dilate(thresh, enhencer, iterations)
        thresh = cv2.erode(thresh, enhencer)
    return thresh


def main():
    plt.figure(0, figsize=(16, 8))
    cap = cv2.VideoCapture('video3.mp4')
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames = [frame] * 3
    while ret:
        ret, frame = cap.read()
        orig_img = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
        frames = frames[1:]
        motion = motion_detector(frames, enhencer=cv2.MORPH_ELLIPSE)
        contours, _ = cv2.findContours(motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounds = [cv2.boundingRect(contour) for contour in contours]
        bounds = bounds_filter(bounds)
        for bound in bounds[:5]:
            x, y, w, h = bound
            cv2.rectangle(orig_img, (x, y), (x+w, y+h), (0, 255, 0))
            # cv2.putText(orig_img, str('%.2f' % score), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        # plt.imshow(frame, cmap='gray')
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 2, 1)
        plt.imshow(orig_img)
        plt.subplot(1, 2, 2)
        plt.imshow(motion, cmap='gray')
        plt.pause(0.001)
        plt.clf()
    cap.release()


if __name__ == '__main__':
    main()

