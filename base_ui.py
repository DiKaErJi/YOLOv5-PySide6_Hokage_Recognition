# -*- coding: utf-8 -*-
# 导入必要的库
from PySide6.QtCore import Qt
import os
import cv2  # OpenCV库，用于视频和图像处理
import sys  # 系统特定的参数和函数
import torch  # PyTorch库，用于深度学习
from PIL import Image  # Pillow库，用于图像格式转换
from PySide6.QtWidgets import QMainWindow, QApplication, QFileDialog  # PySide6库，用于GUI组件
from PySide6.QtGui import QPixmap, QImage  # PySide6库，用于在GUI中处理图像
from PySide6.QtCore import QTimer  # PySide6库，用于计时事件
from main_window import Ui_mainWindow   # 导入由Qt Designer生成的主窗口布局
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = r'D:\software\Anaconda\envs\yolov5\Lib\site-packages\PySide6\plugins\platforms'# 设置Qt平台插件路径的环境变量

# 定义函数将OpenCV图像转换为QImage
def convert2QImage(img):    # 将一个OpenCV格式的图像（一般是NumPy数组）转换为Qt的QImage格式，以便于在PySide的GUI中展示

    # .shape函数获取图像的高度、宽度和通道数，格式为(height, width, channel)
    height, width, channel = img.shape

    # 转换并返回QImage格式，创建QImage对象的参数中width * channel是每行图像数据的字节数，每行字节数 = 图像宽度（像素数） * 每个像素的字节数（通道数）
    # 最后一个参数指定图像的格式
    return QImage(img, width, height, width * channel, QImage.Format_RGB888)

# 定义函数将JPEG图像转换为PNG格式
def convert_jpeg_to_png(file_path):
    if file_path.lower().endswith(('.jpg', '.jpeg')):
        # 打开JPEG图像并转换为PNG格式
        with Image.open(file_path) as img:
            new_file_path = file_path.rsplit('.', 1)[0] + '.png'
            img.save(new_file_path)
            return new_file_path
    return file_path

# 定义主窗口类，继承自QMainWindow和UI布局
class MainWindow(QMainWindow, Ui_mainWindow):

    def __init__(self):     # 初始化 MainWindow 类的实例，加载自定义 YOLOv5 模型，创建和配置一个定时器，初始化视频捕获对象，绑定信号槽函数
        super(MainWindow, self).__init__()  # 初始化父类
        self.setupUi(self)  # 设置UI布局
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowMaximizeButtonHint)
        self.setFixedSize(self.size())  # 设置固定大小

        # self.model 是在 MainWindow 类的构造函数中初始化的模型对象
        # torch.hub.load 是 PyTorch 提供的一个函数，用于从模型库中加载模型。它可以从本地或在线的 PyTorch Hub 加载预训练模型或自定义模型
        # "./":指定当前目录 "./" 作为模型的源目录，PyTorch Hub 将从这个目录中查找和加载模型
        # "custom":这个参数用于指定模型名称。对于自定义模型，可以使用 "custom" 作为标识
        # path="runs/train/exp/weights/best.pt":path 是一个关键字参数，指定模型权重文件的路径,这个路径通常是在训练过程中生成的，表示保存最佳模型权重的文件位置
        self.model = torch.hub.load("./", "custom", path="runs/train/exp/weights/best.pt", source="local")

        # QTimer是PyQt / PySide中的一个类，用于创建计时器
        # self.timer是MainWindow类的一个实例属性，用于保存创建的QTimer对象
        self.timer = QTimer()  # 创建QTimer对象

        # setInterval(1) 是 QTimer 类的方法，用于设置计时器的时间间隔
        self.timer.setInterval(1)  # 设置计时间隔为1毫秒

        # self.video 是 MainWindow 类的一个属性，用来从视频文件或摄像头捕获对象
        self.video = None  # 初始化视频捕获对象

        # 用于定义和连接用户界面控件的信号和槽函数
        self.bind_slots()  # 绑定槽函数到各自的信号

    def image_pred(self, file_path):    # 预测图像中的对象，并返回处理后的QImage

        # 调用 self.model(file_path)，将指定路径的图像文件传递给模型进行预测，返回预测结果
        results = self.model(file_path)  # 从模型获取预测结果

        # 将推理结果的图片数组提出来，而非show
        image = results.render()[0]  # 渲染带预测结果的图像

        # 将numpy接收一个numpy数组作为输入参数,再将数组表示的图像转换为PySide中的QImage对象
        return convert2QImage(image)  # 转换为QImage并返回

    # 方法：处理打开和处理图像的操作
    def open_image(self):
        print("点击了检测图片！")  # 打印信息，用于调试和确认点击事件是否触发
        self.timer.stop()  # 停止计时器，以防止视频仍在播放

        # 打开文件对话框，让用户选择要打开的图像文件
        file_path = QFileDialog.getOpenFileName(self, dir="./datasets/images/train", filter="*.jpg;*.png;*.jpeg")

        # 确保用户选择了文件并获取文件路径
        if file_path[0]:
            file_path = file_path[0]  # 获取文件路径
            print(f"Selected file path: {file_path}")  # 打印文件路径，确保路径正确

            # 如果文件路径不是绝对路径，则转换为绝对路径
            if not os.path.isabs(file_path):
                file_path = os.path.abspath(file_path)

            # 将路径转换为使用反斜杠（Windows路径格式）
            file_path = file_path.replace('/', '\\')

            # 检查文件是否存在
            if not os.path.exists(file_path):
                print(f"File does not exist: {file_path}")  # 如果文件不存在，打印错误信息
                return

            # 检查文件是否可读
            if not os.access(file_path, os.R_OK):
                print(f"File is not readable: {file_path}")  # 如果文件不可读，打印错误信息
                return

            # 将JPEG图像转换为PNG格式，因为模型处理的是PNG格式的图像
            file_path = convert_jpeg_to_png(file_path)

            # 加载图像并显示在输入框中
            pixmap = QPixmap(file_path)  # 创建一个QPixmap对象，用于加载图像文件
            if pixmap.isNull():  # 检查图像是否成功加载
                print(f"Failed to load image from {file_path}")  # 如果图像加载失败，打印错误信息
            else:
                self.input.setPixmap(pixmap)  # 将加载的图像显示在输入框中

            # 对打开的图像进行目标检测，并将处理后的结果显示在输出框中
            qimage = self.image_pred(file_path)  # 调用image_pred函数进行目标检测
            self.output.setPixmap(QPixmap.fromImage(qimage))  # 将处理后的图像显示在输出框中

    def video_pred(self):   # 处理和显示视频帧

        # self.video 是一个cv2.VideoCapture对象，用于打开和读取视频文件
        # read() 方法从视频中读取一帧，并返回两个值：ret表示是否成功读取到帧（True 或 False），frame是表示该帧的图像数据
        ret, frame = self.video.read()  # 从视频读取一帧

        # 视频已经没有更多帧可读的时候停止
        if not ret:  # 如果未读取到帧，停止计时器
            self.timer.stop()
        else:

            # OpenCV 读取的图像默认为BGR格式，而PySide 中常用的是RGB格式。这里使用cv2.cvtColor()方法将BGR格式的帧转换为RGB格式
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # convert2QImage(frame) 将numpy数组格式的帧转换为QImage对象
            # QPixmap.fromImage(convert2QImage(frame))将QImage对象转换为QPixmap对象，用于在界面上显示
            # self.input.setPixmap(...)将转换后的QPixmap对象设置为self.input控件的图像内容，从而显示原始帧
            self.input.setPixmap(QPixmap.fromImage(convert2QImage(frame)))
            results = self.model(frame)  # 获取预测结果
            image = results.render()[0]  # 渲染带预测结果的帧
            self.output.setPixmap(QPixmap.fromImage(convert2QImage(image)))  # 显示处理后的帧

    def open_video(self):   # 处理打开和处理视频的操作

        print("点击了检测视频！")
        # 打开文件对话框选择视频文件
        file_path = QFileDialog.getOpenFileName(self, dir="./datasets", filter="*.mp4")
        if file_path[0]:  # 如果选择了文件
            file_path = file_path[0]  # 获取文件路径
            self.video = cv2.VideoCapture(file_path)  # 打开视频文件
            self.timer.start()  # 启动计时器

    def quit(self):     # 处理退出程序操作

        print("点击了退出程序！")
        sys.exit(0)  # 退出应用程序

    # 方法：将槽函数绑定到各自的信号（事件）
    def bind_slots(self):

        self.det_image.clicked.connect(self.open_image)  # 将open_image绑定到detect_image按钮点击事件
        self.det_video.clicked.connect(self.open_video)  # 将open_video绑定到detect_video按钮点击事件
        self.exit.clicked.connect(self.quit)  # 将quit绑定到exit按钮点击事件

        # timeout 是 QTimer 的信号，表示计时器超时事件
        self.timer.timeout.connect(self.video_pred)  # 将video_pred绑定到计时器超时事件

# 应用程序的主入口
if __name__ == "__main__":
    app = QApplication(sys.argv)  # 使用命令行参数创建QApplication对象
    window = MainWindow()  # 创建主窗口实例
    window.show()  # 显示主窗口
    app.exec()  # 执行应用程序事件循环

# 这是一个基于PyQt的GUI应用程序，用于使用YOLOv5模型进行图像和视频的目标检测。它提供了打开和处理图像和视频的功能，并将原始和处理后的结果并排显示。
# 脚本设置了主窗口，加载训练好的模型，将UI元素绑定到各自的事件处理函数，并使用YOLOv5模型处理图像和视频任务。
