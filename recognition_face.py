import cv2
import os
import gain_face
from face_train import Model, Dataset


def main():
    judge = False
    while True:
        print("是否录入人脸信息(Yes or No)?，请输入英文名")
        input_ = input()
        if input_ == 'Yes':
            # 员工姓名(要输入英文)
            new_user_name = input("请输入您的姓名：")
            print("请看摄像头！")
            judge = True
            # 采集员工图像的数量自己设定，越多识别准确度越高，但训练速度贼慢
            window_name = 'Information Collection'  # 图像窗口
            camera_id = 0  # 相机的ID号
            images_num = 100  # 采集图片数量
            path = './FaceData/' + new_user_name  # 图像保存位置
            gain_face.CatchPICFromVideo(window_name, camera_id, images_num, path, new_user_name)
        elif input_ == 'No':
            break
        else:
            print("错误输入，请输入Yes或者No")
    # 加载模型
    if judge == True:

        user_num = len(os.listdir('./FaceData/'))

        dataset = Dataset('./FaceData/')

        dataset.load()

        model = Model()

        # 先前添加的测试build_model()函数的代码
        model.build_model(dataset, nb_classes=user_num)

        # 测试训练函数的代码
        model.train(dataset)

        model.save_model(file_path='./model/aggregate.face.model.h5')

    else:
        model = Model()

        model.load_model(file_path='./model/aggregate.face.model.h5')

    # 框住人脸的矩形边框颜色
    color = (255, 255, 255)

    # 捕获指定摄像头的实时视频流
    cap = cv2.VideoCapture(0)

    # 人脸识别分类器本地存储路径
    cascade_path = "./haarcascade_frontalface_alt2.xml"

    # 循环检测识别人脸

    while True:
        ret, frame = cap.read()  # 读取一帧视频

        if ret is True:
            # 图像灰化，降低计算复杂度
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            continue
        # 使用人脸识别分类器，读入分类器
        cascade = cv2.CascadeClassifier(cascade_path)

        # 利用分类器识别出哪个区域为人脸
        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=2, minSize=(32, 32))

        for (x, y, w, h) in faceRects:
            # 截取脸部图像提交给模型识别这是谁
            image = frame[y: y + h, x: x + w]

            faceID = model.face_predict(image)

            print(faceID)

            cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=1)

            for i in range(len(os.listdir('./FaceData/'))):

                if i == faceID:
                    # 文字提示是谁
                    cv2.putText(frame, os.listdir('./FaceData/')[i], (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                color, 2)
        cv2.imshow("recognition! press 'Q' to quit", frame)
        # 等待10毫秒看是否有按键输入
        k = cv2.waitKey(10)
        # 如果输入q则退出循环
        if k & 0xFF == ord('q'):
            break
    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()