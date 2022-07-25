# 数字图像处理系统项目报告
#### 摘要
这个项目的华东师范大学软件工程学院数字图像处理的期末完整结课项目。由本人单独完成。这个项目主要实现了有以下的六个功能：

1.数字图像基础  2.边缘检测  3.图像增强  4.图像修复  5.图像分割  6.人脸识别

其中前五个功能基于opencv+python所实现的，最后一个人脸识别是结合了机器学习的知识，使用tensorflow+opencv+python
构建了卷积神经网络模型来实现的多人人脸识别功能。

#### 环境配置

1. python 3.6
2. tensorflow 2.1.0(安装cpu版本) 
3. keras 2.3.1 
4. scikit-learn 0.20.3 
5. opencv-python 任意版本
6. numpy
7. pycharm 2021.3.3

#### 安装教程

不做赘述，利用anaconda，搭建一个新的虚拟环境来进行以上环境的一个一个的配置

#### 使用说明

执行项目中的main()函数来进行整个项目的运行，根据提示的文字来进行一系列的操作

FaceData为存储人脸数据的文件，用来存放数据集

model为存储人脸识别模型的文件，用来存放训练完的CNN模型

Wangwenhai 为本人的一段视频数据，用于收集数据

#### 正文

接下来将会对我如何实现的这系列功能做详细的介绍

#### 数字图像基础
图像的仿射变化：
    使用cv2.warpAffine()函数，来实现图像的仿射变化。仿射变换可以通
过一系列的原子变换的复合来实现包括：平移(Translation)、缩放(Scale)、翻转(Flip)、
旋转(Rotation)和错切(Shear)这些功能。 至于如何具体实现这些功能，则要取决于2*3的
变换矩阵 M的具体内容

图形的扩展缩放、平移、旋转：
    这个功能的实现结合的图像的仿射变化，其中缩放利用了cv2,resize()函数来实现对图像的缩放，
cv2.warpAffine()实现了图像的平移和利用了 cv2.getRotationMatrix2D()这个函数来构造旋转
是的变换矩阵来用于图像的仿射变化。是仿射变化的拓展和更多的实际应用。本功能实现的是图像的x轴左移
10个像素，y轴下移30个。之后进行旋转中心坐标为处理后图片长宽的一半，旋转角度为45度，缩放因子为1的旋转。

#### 边缘检测
边缘检测目的是找到图像中亮度变化剧烈的像素点构成的集合。由此构建出来的轮廓来更好的实现精准的测量和
定位。图像边缘有两个要素：方向和幅度。沿着边缘走向的像素值变化比较平缓；而沿着垂直于边缘的走向，
像素值则变化得比较大。因此，我们采用一阶和二阶导数来描述和检测边缘。 我实现了以下四个算子：

一阶微分算子：Roberts,Sobel算子

二阶微分算子：Laplacian，Laplacian-gauss算子

#### 图像增强

图像增强可以分为四个模块-空域的锐化，空域的平滑，频域的锐化，频域的平滑下面将分别介绍实现的方式

空域的锐化：锐化的作用一般是用来加强图像的轮廓和边缘，其本质是高通滤波器。锐化的结果是增强了边缘。
但是图像本身的层次和亮度已基本丢失。这里使用的方式与边缘检测雷同。
我使用的一阶微分算子：Roberts算子，Sobel算子，Prewitt算子
二阶微分算子：Laplacian算子

空域的平滑：图像平滑的目的是改善图像质量，尽量消除噪声对图像带来的影响，其本质是低通滤波。图像的
空域平滑实现起来很简单，只要将原图中的每一个点的灰度与它周围点的灰度进行加权和平均，作为新图中对
应点的灰度，就能实现滤波的效果。以下我使用了三种滤波器，分别是： 中值滤波器，均值滤波器，高斯滤波器
其中高斯滤波的效果最好，由于引入了加权系数，平滑的效果也是最好的。

频域的锐化：从频谱分析的角度看, 图像都是由决定图像反差的低频信号和决定图像细节的高频信号组
成。但数字化图像中高频信号部分总是掺杂有一定程度的噪声。因此, 在频率域中进行图像的锐
化处理实质上是加强需要的高频分量,并必须考虑到要在锐化图像的同时抑制噪声。频率域中滤波的数学表达式可写为

G(u,v)=H(u,v)⋅F(u,v) 

上式中,F(u,v)是原始图像的Fouirer频谱,G(u,v)是锐化后图像的Fouirer频谱,H(u,v)是滤波器的转移函数。
我使用的三种滤波器： 1.理想高通滤波 2.巴特沃斯高通滤波 3.指数高通滤波器

频域的平滑：再频谱分析中，噪声往往有着较丰富的高频分量，此时就需要低通滤波来降噪。我是用了三种滤波器，
1.理想低通滤波 2.巴特沃斯低通滤波 3.指数低通滤波器
总而言之，平滑就是降噪，锐化就是突出边缘

#### 图像修复
图像就是利用那些已经被破坏的区域的边缘，根据这些图像留下的信息去推断被破坏的区域的信息内容，然后对
破坏区进行填补 ，以达到图像修补的目的。这里我是用了两种方式，一种是opencv库中的cv2.inpaint()函数。
第二种则是模仿人类视觉系统的Retinex算法，适合对光照条件不好，有迷雾的图片做修复。下面是详细介绍

1.inpaint函数： 在opencv种，有dst = cv2.inpaint（src，mask, inpaintRadius，flags）其中，
其中，src是原图像，inpaintMask是掩膜，是需要修复的部分，inpaintRadius是算法需要考虑的每个点的圆形
邻域的半径，flags则是使用的算法INPAINT_NS和INPAINT_TELEA。

2.Retinex算法：Retinex理论的基础理论是物体的颜色是由物体对长波（红色）、中波（绿色）、 短波（蓝色）
光线的反射能力来决定的，而不是由反射光强度的绝对值来决定的，物体的色彩不受光照非均匀性的影响，具有一致性，
即retinex是以色感一致性（颜色恒常性）为基础的。下面介绍我实现的四类Retinex算法：单尺度Retinex算法，
MSR改进成多尺度加权平均的MSR算法，彩色恢复多尺度MSRCR算法和色彩增益加权的AutoMSRCR算法。

SSR：先将图像进行log变换，然后将log图像进行高斯模糊，最后利用原图和模糊之后的log图像做差分

MSR：原始图像进行三次SSR，高斯模糊选择15，80，200作为高斯模糊sigma参数，对三次的SSR结果做平均即为MSR图像

MSRCR：autoMSRCR：对多尺度MSR结果做了色彩平衡，归一化，增益和偏差线性加权

#### 图像分割
图像分割的实现使用深度学习的知识会有更好的实现效果，这里我没有使用深度学习。我是使用 OpenCV 函数filter2D
执行一些拉普拉斯滤波以进行图像锐化来突出细节，distanceTransform 以获得二值图像的派生(derived)表示，其中
每个像素的值被替换为其到最近背景像素的距离，并执行一些图像形态学操作来提取峰值，最后使用watershed（分水岭算法）
将图像中的对象与背景隔离。最终实现了图像的分割。

#### 人脸识别
（写在前面，人脸识别的数据必须是两个人以上，只录入一个人的图像是无法开始训练的。训练的模型保存在model中，若不想
使用电脑摄像头来拍摄图像，可以修改gain_face中cv2.VideoCapture()的参数来从硬盘中读取数据）

这个部分使用了机器学习的知识，利用CNN搭建了一个人脸识别模型（只能识别多人，单人不行），下面介绍详细的流程：

1.图片的获取：这里我调用了电脑自带的摄像机来获取数据集，每个人拍摄100张照片（缩短训练时间，可以在recognition_face.py）
中修改images_num来改变每个人的照片数量。使用haarcascade_frontalface_alt2人脸识别器来捕捉人脸图片保存在Facedata，
并且转为灰度图来 提高计算的效率。（考虑到灰度图可能需要增强，可以在gain_face.py最后面加上增强的代码）

2.数据集的构建：这里将所有获取的图像的大小都设计为128*128，并且使用在Facedata这个文件中人脸数据文件夹的次序作为唯一标签。比
如 Wangwenhai这个文件是目录下的第1个文件，那么Wangwenhai里的图像的标签就是1

3.人脸数据的训练：我构建了一个18层的CNN模型，用来训练数据，并且使用了adam来作为模型的优化器，由于我构建的是多分类模型。故使用
binary_crossentropy作为损失函数。同时我还定义一个生成器用于数据提升，其返回一个生成器对象datagen来强化学习。最后，由face_predict
函数来返回预测的结果。

4.人脸识别:利用模型返回的预测值来对摄像头扑捉到的图像做出预测

#### 总结
通过长达两个星期的学习，我对计算机视觉这方面的知识有了更为深刻的理解，对opencv的使用更加熟练。而最令我收获最多的是我完成了我的第一
个人脸识别的项目，算是对机器学习有了浅浅的入门。总而言之，这个项目由我单独完成，我从这个课程收获颇多。

