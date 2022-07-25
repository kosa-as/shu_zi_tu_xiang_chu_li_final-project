import os
import cv2
import numpy as np
import sys
import json
def inpaint():
    cap = cv2.VideoCapture(0)
    ok, src = cap.read()
    # cap默认拍摄640*480的照片,现在破坏图片
    for i in range(200, 300):
        src[200, i] = 255
        src[200 + 1, i] = 255
        src[200 - 1, i] = 255
    for i in range(150, 250):
        src[i, 250] = 255
        src[i, 250 + 1] = 255
        src[i, 250 - 1] = 255
    cap.release()
    if ok:
        img = src
        height = img.shape[0]
        width = img.shape[1]
        #选择要修复的区域用paint表示
        paint = np.zeros((height, width, 1), np.uint8)
        for i in range(200, 300):
            paint[200, i] = 255
            paint[200 + 1, i] = 255
            paint[200 - 1, i] = 255
        for i in range(100, 300):
            paint[i, 250] = 255
            paint[i, 250 + 1] = 255
            paint[i, 250 - 1] = 255
        result = cv2.inpaint(img, paint, 3, cv2.INPAINT_TELEA)
        img = np.hstack([src, result])
        cv2.imshow("source and result", img)
        # 等待关闭
        cv2.waitKey(0)


def singleScaleRetinex(img, sigma):
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))

    return retinex


def multiScaleRetinex(img, sigma_list):
    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        retinex += singleScaleRetinex(img, sigma)

    retinex = retinex / len(sigma_list)

    return retinex


def colorRestoration(img, alpha, beta):
    img_sum = np.sum(img, axis=2, keepdims=True)

    color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))

    return color_restoration


def simplestColorBalance(img, low_clip, high_clip):
    total = img.shape[0] * img.shape[1]
    for i in range(img.shape[2]):
        unique, counts = np.unique(img[:, :, i], return_counts=True)
        current = 0
        for u, c in zip(unique, counts):
            if float(current) / total < low_clip:
                low_val = u
            if float(current) / total < high_clip:
                high_val = u
            current += c

        img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)

    return img


def MSRCR(img, sigma_list, G, b, alpha, beta, low_clip, high_clip):
    img = np.float64(img) + 1.0

    img_retinex = multiScaleRetinex(img, sigma_list)

    img_color = colorRestoration(img, alpha, beta)
    img_msrcr = G * (img_retinex * img_color + b)

    for i in range(img_msrcr.shape[2]):
        img_msrcr[:, :, i] = (img_msrcr[:, :, i] - np.min(img_msrcr[:, :, i])) / \
                             (np.max(img_msrcr[:, :, i]) - np.min(img_msrcr[:, :, i])) * \
                             255

    img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255))
    img_msrcr = simplestColorBalance(img_msrcr, low_clip, high_clip)

    return img_msrcr


def automatedMSRCR(img, sigma_list):
    img = np.float64(img) + 1.0

    img_retinex = multiScaleRetinex(img, sigma_list)

    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break

        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break

        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)

        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255

    img_retinex = np.uint8(img_retinex)

    return img_retinex


def MSRCP(img, sigma_list, low_clip, high_clip):
    img = np.float64(img) + 1.0

    intensity = np.sum(img, axis=2) / img.shape[2]

    retinex = multiScaleRetinex(intensity, sigma_list)

    intensity = np.expand_dims(intensity, 2)
    retinex = np.expand_dims(retinex, 2)

    intensity1 = simplestColorBalance(retinex, low_clip, high_clip)

    intensity1 = (intensity1 - np.min(intensity1)) / \
                 (np.max(intensity1) - np.min(intensity1)) * \
                 255.0 + 1.0

    img_msrcp = np.zeros_like(img)

    for y in range(img_msrcp.shape[0]):
        for x in range(img_msrcp.shape[1]):
            B = np.max(img[y, x])
            A = np.minimum(256.0 / B, intensity1[y, x, 0] / intensity[y, x, 0])
            img_msrcp[y, x, 0] = A * img[y, x, 0]
            img_msrcp[y, x, 1] = A * img[y, x, 1]
            img_msrcp[y, x, 2] = A * img[y, x, 2]

    img_msrcp = np.uint8(img_msrcp - 1.0)

    return img_msrcp


def retinex():
    with open('config.json', 'r') as f:
        config = json.load(f)
    cap = cv2.VideoCapture(0)
    ok, src = cap.read()
    img = src
    cap.release()
    if ok:
        print("选择类别：1..彩色恢复多尺度Retinex（MSRCR） 2.彩色恢复多尺度Retinex（MSRCP） 3.彩色恢复多尺度Retinex(AMSRCR)")
        print("#Retinex对清晰的图像处理效果不佳，适合用来处理光线不好，有雾等这些类的图片#")
        myinput = input()
        if myinput == '1':
            img_msrcr = MSRCR(
                img,
                config['sigma_list'],
                config['G'],
                config['b'],
                config['alpha'],
                config['beta'],
                config['low_clip'],
                config['high_clip']

            )
            img = np.hstack([src, img_msrcr])
            cv2.imshow('resourse and result', img)
            cv2.waitKey(0)
        elif myinput == '3':
            img_amsrcr = automatedMSRCR(
                    img,
                    config['sigma_list']
            )
            img = np.hstack([src, img_amsrcr])
            cv2.imshow('resourse and result', img)
            cv2.waitKey(0)
        elif myinput == '2':
            img_msrcp = MSRCP(
                    img,
                    config['sigma_list'],
                    config['low_clip'],
                    config['high_clip']
            )
            img = np.hstack([src, img_msrcp])
            cv2.imshow('resourse and result', img)
            cv2.waitKey(0)
        else:
            print("Wrong input!")

def main():
    print("选择图像修复的算法 1.opencv中inpaint图像修复 2.模仿人类视觉系统的Retinex算法")
    myinput = input()
    if myinput == '1':
        inpaint()
    elif myinput == '2':
        retinex()
    else:
        print("Wrong input!")