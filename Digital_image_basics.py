import cv2
import sys
import os
import numpy as np
#仿射变化
def Affinevariation():
    cap = cv2.VideoCapture(0)
    ok, src = cap.read()
    cap.release()
    if ok:
        # 图像放缩
        src = cv2.resize(src, (256, 256))
        # 获取图像shape
        rows, cols = src.shape[: 2]
        ########Begin########
        #设置图像仿射变化矩阵
        post1 = np.float32([[50, 50], [200, 50], [50, 200]])
        post2 = np.float32([[10, 100], [200, 50], [100, 250]])
        M = cv2.getAffineTransform(post1, post2)
        # 图像仿射变换
        result = cv2.warpAffine(src, M, (rows, cols))
        img = np.hstack([src, result])
        cv2.imshow("resourse and result", img)
        # 等待关闭
        cv2.waitKey(0)
#图形扩展缩放、平移、旋转
def Graphics_extend_scale_pan_rotate():
    cap = cv2.VideoCapture(0)
    ok, src = cap.read()
    cap.release()
    if ok:
        # 图像放缩
        img = src
        l, w, h = img.shape
        # 放大图像至原来的两倍，使用双线性插值法

        cv2.resize(img, (0, 0), 2, 2, cv2.INTER_LINEAR)

        height, width, channel = img.shape
        # 构建移动矩阵,x轴左移 10 个像素，y轴下移 30 个

        M = np.float32([[1, 0, 10], [0, 1, 30]])

        img = cv2.warpAffine(img, M, (width, height))
        # 构建矩阵，旋转中心坐标为处理后图片长宽的一半，旋转角度为45度，缩放因子为1

        M = cv2.getRotationMatrix2D((width / 2, height / 2), 45, 1)

        dst = cv2.warpAffine(img, M, (width, height))

        img = np.hstack([src, dst])

        cv2.imshow("resourse and result", img)
        # 等待关闭
        cv2.waitKey(0)
def main():
    print("1.仿射变化 2.图形扩展缩放、平移、旋转")
    myinput = input()
    if myinput == '1':
        Affinevariation()
    elif myinput == '2':
        Graphics_extend_scale_pan_rotate()
    else:
        print("wrong input!")


