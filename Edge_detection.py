import os
import cv2
import numpy as np
import sys
def robs():
    cap = cv2.VideoCapture(0)
    ok, src = cap.read()
    cap.release()
    #src = cv2.imread('./zero.jpg')
    if ok:
        img = src
        ########## Begin ##########
        # 1. 灰度化处理图像
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 2. Roberts算子
        kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
        kernely = np.array([[0, -1], [1, 0]], dtype=int)
        # 3. 卷积操作
        x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
        y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)
        # 4. 数据格式转换
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        ########## End ##########
        print(src.shape)
        #单通道转为三通道
        Roberts=cv2.cvtColor(Roberts, cv2.COLOR_GRAY2RGB)

        img = np.hstack([src, Roberts])
        cv2.imshow("resourse and result", img)
        # 等待关闭
        cv2.waitKey(0)
def sob():
    cap = cv2.VideoCapture(0)
    ok, src = cap.read()
    cap.release()
    if ok:
        img = src
        ########## Begin ##########
        # 1. 灰度化处理图像
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 2. 求Sobel 算子
        x = cv2.Sobel(grayImage, cv2.CV_16S, 1, 0)  # 对x求一阶导
        y = cv2.Sobel(grayImage, cv2.CV_16S, 0, 1)  # 对y求一阶导
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        ########## End ##########
        Sobel = cv2.cvtColor(Sobel, cv2.COLOR_GRAY2RGB)
        img = np.hstack([src, Sobel])
        cv2.imshow("resourse and result", img)
        # 等待关闭
        cv2.waitKey(0)
def lap():
    cap = cv2.VideoCapture(0)
    ok, src = cap.read()
    cap.release()
    if ok:
        img = src
        ########## Begin ##########
        # 1. 灰度化处理图像
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 2. 高斯滤波
        grayImage= cv2.GaussianBlur(grayImage, (5, 5), 0)
        # 3. 拉普拉斯算法
        dst = cv2.Laplacian(grayImage, cv2.CV_16S, ksize=3)
        # 4. 数据格式转换
        Laplacian = cv2.convertScaleAbs(dst)

        Laplacian = cv2.cvtColor(Laplacian, cv2.COLOR_GRAY2RGB)
        ########## End ##########
        img = np.hstack([src, Laplacian])
        cv2.imshow("resourse and result", img)
        # 等待关闭
        cv2.waitKey(0)
def _log():
    cap = cv2.VideoCapture(0)
    ok, src = cap.read()
    cap.release()
    if ok:
        img = src
        ########## Begin ##########
        # 1. 灰度转换
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 2. 边缘扩充处理图像并使用高斯滤波处理该图像
        image = cv2.copyMakeBorder(img,2,2,2,2,borderType=cv2.BORDER_REPLICATE)
        image = cv2.copyMakeBorder(img,2,2,2,2,borderType=cv2.BORDER_REPLICATE)
        image = cv2.GaussianBlur(image,(3,3),0)
        # 3. 使用Numpy定义LoG算子
        m1=np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]])
        image1=np.zeros(image.shape,dtype='float')
        # 4. 卷积运算
        # 为了使卷积对每个像素都进行运算，原图像的边缘像素要对准模板的中心。
        # 由于图像边缘扩大了2像素，因此要从位置2到行(列)-2
        for i in range(2,image.shape[0]-2):
            for j in range(2,image.shape[1]-2):
               image1[i,j]=np.sum(m1*image[i-2:i+3,j-2:j+3,1])
        # 5. 数据格式转换
        image1 = cv2.convertScaleAbs(image1)

        image1 = cv2.resize(image1, (img.shape[1], img.shape[0]))
        #print(img.shape)
        #print(image1.shape)
        img = np.hstack([src, image1])
        cv2.imshow("resourse and result", img)
        # 等待关闭
        cv2.waitKey(0)
def main():
    print("1.Roberts算子 2.Sobel算子 3.Laplacian算子 4.log边缘算子")
    myinput = input()
    if myinput == '1':
        robs()
    elif myinput == '2':
        sob()
    elif myinput == '3':
        lap()
    elif myinput == '4':
        _log()
    else:
        print("wrong input!")

