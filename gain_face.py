import cv2
import os
import sys
import numpy as np
from createfold import CreateFolder
def CatchPICFromVideo (window_name, camera_idx, catch_pic_num, path_name, usr_name):

        #camera_idx代表摄像头编号，0即为系统默认

        #检查输入路径是否存在——不存在就创建

        CreateFolder(path_name)

        cv2.namedWindow(window_name)

        #cap = cv2.VideoCapture('./Wangwenhai/Wangwenhai.mp4')
        # 设置分辨率
        cap = cv2.VideoCapture(0)
        cap.set(3, 1920)

        cap.set(4, 1080)

        # 告诉OpenCV使用人脸识别分类器

        classfier = cv2.CascadeClassifier("./haarcascade_frontalface_alt2.xml")

        #记录已拍摄的照片数目

        num = 0

        while True:

            ok, frame = cap.read()  # 读取一帧数据

            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #将当前桢图像转换成灰度图像

            # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数

            faceRects = classfier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=2, minSize=(32, 32))

            for (x, y, w, h) in faceRects:
                #单独框出每一张人脸

                #将当前帧保存为图片

                img_name = '%s/%d.jpg' % ("./FaceData/"+usr_name, num)

                #保存灰度人脸图

                cv2.imwrite(img_name, gray[y:y+h, x:x+w])

                num += 1

                #画出矩形框的时候稍微比识别的脸大一圈

                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10),(0, 255, 0),1)

                #显示当前捕捉到了多少人脸图片

                font = cv2.FONT_HERSHEY_SIMPLEX

                cv2.putText(frame, 'num:%d' % (num), (x + 30, y + 30), font, 1, (255, 0, 255), 1)

                # 超过指定最大保存数量结束程序

            cv2.imshow(window_name, frame)
            if num >= catch_pic_num:
                break
            #按键盘‘Q’中断采集
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        # 释放摄像头并销毁所有窗口
        print("拍摄完成，训练中......")
        cap.release()
        cv2.destroyAllWindows()
        """
        for dir_item in os.listdir(path_name):
            full_path = path_name + '\\' + dir_item
            if os.path.isdir(full_path):
                read_path(full_path)
            else:
                # 判断是人脸照片
                if dir_item.endswith('.jpg'):
                ##################################
                此处可对拍摄的灰度图进行图像增强，如有需要
                ##################################
        """

