import numpy as np
import cv2
import os
import sys
import Digital_image_basics
import Edge_detection
import Image_enhancement
import Image_repair
import Image_segmentation
import recognition_face
if __name__ == '__main__':
    print("选择你要实现的功能")

    print("1.数字图像基础 2.边缘检测 3.图像增强 4.图像修复 5.图像分割 6.人脸识别")

    myinput = input()

    if myinput == '1':
        Digital_image_basics.main()
    elif myinput == '2':
        Edge_detection.main()
    elif myinput == '3':
        Image_enhancement.main()
    elif myinput == '4':
        Image_repair.main()
    elif myinput == '5':
        Image_segmentation.main()
    elif myinput == '6':
        recognition_face.main()
    else:
        print("wrong input!")
