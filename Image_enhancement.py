import sys
import cv2
import os
import numpy as np
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
def pre():
     cap = cv2.VideoCapture(0)
     ok, src = cap.read()
     cap.release()

     if ok:
          img = src

          ########## Begin ##########
          # 1. 灰度化处理图像

          grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
          kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
          kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
          x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
          y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)
          # 转uint8
          absX = cv2.convertScaleAbs(x)
          absY = cv2.convertScaleAbs(y)
          Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

          Prewitt = cv2.cvtColor(Prewitt, cv2.COLOR_GRAY2RGB)
          ########## End ##########
          img = np.hstack([src, Prewitt])
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
def Blur():
    cap = cv2.VideoCapture(0)
    ok, src = cap.read()
    cap.release()
    if ok:
        img = src
        #source = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = cv2.blur(img, (3, 3))
        img = np.hstack([src, result])
        cv2.imshow("resourse and result", img)
        # 等待关闭
        cv2.waitKey(0)
def medianblur():
    cap = cv2.VideoCapture(0)
    ok, src = cap.read()
    cap.release()
    if ok:
        img = src
        result = cv2.medianBlur(img, 5)
        img = np.hstack([src, result])
        cv2.imshow("resourse and result", img)
        # 等待关闭
        cv2.waitKey(0)
def gauss():
    cap = cv2.VideoCapture(0)
    ok, src = cap.read()
    cap.release()
    if ok:
        img = src
        result = cv2.GaussianBlur(img, (3, 3), 0)
        img = np.hstack([src, result])
        cv2.imshow("resourse and result", img)
        # 等待关闭
        cv2.waitKey(0)


def ideal_high_filter(img, D0):
    """
    生成一个理想高通滤波器（并返回）
    """
    h, w = img.shape[:2]
    filter_img = np.zeros((h, w))
    u = np.fix(h / 2)
    v = np.fix(w / 2)
    for i in range(h):
        for j in range(w):
            d = np.sqrt((i - u) ** 2 + (j - v) ** 2)
            filter_img[i, j] = 0 if d < D0 else 1
    return filter_img


def butterworth_high_filter(img, D0, rank):
    """
        生成一个Butterworth高通滤波器（并返回）
    """
    h, w = img.shape[:2]
    filter_img = np.zeros((h, w))
    u = np.fix(h / 2)
    v = np.fix(w / 2)
    for i in range(h):
        for j in range(w):
            d = np.sqrt((i - u) ** 2 + (j - v) ** 2)
            filter_img[i, j] = 1 / (1 + (D0 / d) ** (2 * rank))
    return filter_img


def exp_high_filter(img, D0, rank):
    """
        生成一个指数高通滤波器（并返回）
    """
    h, w = img.shape[:2]
    filter_img = np.zeros((h, w))
    u = np.fix(h / 2)
    v = np.fix(w / 2)
    for i in range(h):
        for j in range(w):
            d = np.sqrt((i - u) ** 2 + (j - v) ** 2)
            filter_img[i, j] = np.exp((-1) * (D0 / d) ** rank)
    return filter_img


def filter_use(img, filter):
    """
    将图像img与滤波器filter结合，生成对应的滤波图像
    """
    # 首先进行傅里叶变换
    f = np.fft.fft2(img)
    f_center = np.fft.fftshift(f)
    # 应用滤波器进行反变换
    S = np.multiply(f_center, filter)  # 频率相乘——l(u,v)*H(u,v)
    f_origin = np.fft.ifftshift(S)  # 将低频移动到原来的位置
    f_origin = np.fft.ifft2(f_origin)  # 使用ifft2进行傅里叶的逆变换
    f_origin = np.abs(f_origin)  # 设置区间
    f_origin = f_origin / np.max(f_origin.all())
    return f_origin



def ideal_high():
    cap = cv2.VideoCapture(0)
    ok, frame = cap.read()
    src = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cap.release()
    if ok:
        img = src
        ideal_filter = ideal_high_filter(img, D0=40)
        ideal_img = filter_use(img, ideal_filter)
        cv2.imshow("resourse", src)
        cv2.imshow("result", ideal_img)
        # 等待关闭
        cv2.waitKey(0)



def butterworth_high():
    cap = cv2.VideoCapture(0)
    ok, frame = cap.read()
    src = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cap.release()
    if ok:
        img = src
        butterworth_filter = butterworth_high_filter(img, D0=40, rank=2)
        butterworth_img = filter_use(img, butterworth_filter)
        cv2.imshow("resourse", src)
        cv2.imshow("result", butterworth_img)
        # 等待关闭
        cv2.waitKey(0)


def exp_high():
    cap = cv2.VideoCapture(0)
    ok, frame = cap.read()
    src = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cap.release()
    if ok:
        img = src
        exp_filter = exp_high_filter(img, D0=40, rank=2)
        exp_img = filter_use(img, exp_filter)
        cv2.imshow("resourse", src)
        cv2.imshow("result", exp_img)
        # 等待关闭
        cv2.waitKey(0)


def get_gray():
    cap = cv2.VideoCapture(0)
    ok, frame = cap.read()
    src = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cap.release()
    return src

def filter(img, D0, type, filter, N=2):
    '''
    频域滤波器
    Args:
        img: 灰度图片
        D0: 截止频率
        N: butterworth的阶数(默认使用二阶)
        type: lp-低通 hp-高通
        filter:butterworth、ideal、Gaussian即巴特沃斯、理想、高斯滤波器
    Returns:
        imgback：滤波后的图像
    '''
    # 离散傅里叶变换
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    # 中心化
    dtf_shift = np.fft.fftshift(dft)

    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)  # 计算频谱中心
    mask = np.zeros((rows, cols, 2))  # 生成rows行cols列的二维矩阵

    for i in range(rows):
        for j in range(cols):
            D = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2) # 计算D(u,v)
            if (filter.lower() == 'butterworth'):  # 巴特沃斯滤波器
                if (type == 'lp'):
                    mask[i, j] = 1 / (1 + (D / D0) ** (2 * N))
                elif (type == 'hp'):
                    mask[i, j] = 1 / (1 + (D0 / D) ** (2 * N))
                else:
                    assert ('type error')
            elif (filter.lower() == 'ideal'):  # 理想滤波器
                if (type == 'lp'):
                    if (D <= D0):
                        mask[i, j] = 1
                elif (type == 'hp'):
                    if (D > D0):
                        mask[i, j] = 1
                else:
                    assert ('type error')
            elif (filter.lower() == 'gaussian'):  # 高斯滤波器
                if (type == 'lp'):
                    mask[i, j] = np.exp(-(D * D) / (2 * D0 * D0))
                elif (type == 'hp'):
                    mask[i, j] = (1 - np.exp(-(D * D) / (2 * D0 * D0)))
                else:
                    assert ('type error')

    fshift = dtf_shift * mask

    f_ishift = np.fft.ifftshift(fshift)

    img_back = cv2.idft(f_ishift)

    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])  # 计算像素梯度的绝对值

    img_back = np.abs(img_back)

    return img_back

def main():
     print("1.空域的锐化 2.空域的平滑 3.频域的锐化 4.频域的平滑")
     cin = input()
     if cin == '1':
         print("选择锐化的方式：1.Roberts锐化 2.Sobel锐化 3.Laplacian锐化 4.Prewitt锐化")
         myinput = input()
         if myinput == '1':
              robs()
         elif myinput == '2':
              sob()
         elif myinput == '3':
              lap()
         elif myinput == '4':
              pre()
         else:
              print("wrong input!")
     elif cin == '2':
        print("选择滤波器： 1.均值滤波 2.中值滤波 3.高斯滤波")
        myinput = input()
        if myinput == '1':
            Blur()
        elif myinput == '2':
            medianblur()
        elif myinput == '3':
            gauss()
        else:
            print("Wrong input！")
     elif cin == '3':
         print("选择滤波器： 1.理想高通滤波 2.巴特沃斯高通滤波 3.指数高通滤波器")
         myinput = input()
         if myinput == '1':
             ideal_high()
         elif myinput == '2':
             butterworth_high()
         elif myinput == '3':
             exp_high()
         else:
             print("Wrong input!")
     elif cin == '4':
         print("选择滤波器： 1.理想低通滤波 2.巴特沃斯低通滤波 3.指数低通滤波器")
         myinput = input()
         if myinput == '1':
             src = get_gray()
             img = filter(src, 30, type='lp', filter='ideal')
             cv2.imshow("resourse", src)
             cv2.imshow("result", img)
             # 等待关闭
             cv2.waitKey(0)
         elif myinput == '2':
             src = get_gray()
             img = filter(src, 30, type='lp', filter='butterworth')
             cv2.imshow("resourse", src)
             cv2.imshow("result", img)
             # 等待关闭
             cv2.waitKey(0)
         elif myinput == '3':
             src = get_gray()
             img = filter(src, 30, type='lp', filter='gaussian')
             cv2.imshow("resourse", src)
             cv2.imshow("result", img)
             cv2.waitKey(0)
         else:
             print("Wrong input!")
     else:
        print("Wrong input!")

