# -*- coding: utf-8 -*-
import os

import cv2
import sys
import torch
from PySide6.QtWidgets import QMainWindow, QApplication, QFileDialog
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import QTimer

from main_window import Ui_MainWindow
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = r'D:\software\Anaconda\envs\yolov5\Lib\site-packages\PySide6\plugins\platforms'

def convert2QImage(img):
    height, width, channel = img.shape
    return QImage(img, width, height, width * channel, QImage.Format_RGB888)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.model = torch.hub.load("./", "custom", path="runs/train/exp/weights/best.pt", source="local")
        self.timer = QTimer()
        self.timer.setInterval(1)
        self.video = None
        self.bind_slots()

    def image_pred(self, file_path):
        results = self.model(file_path)
        image = results.render()[0]
        return convert2QImage(image)

    def open_image(self):
        print("点击了检测图片！")
        self.timer.stop()
        file_path = QFileDialog.getOpenFileName(self, dir="./datasets/images/train", filter="*.jpg;*.png;*.jpeg")
        if file_path[0]:
            file_path = file_path[0]
            qimage = self.image_pred(file_path)
            self.input.setPixmap(QPixmap(file_path))
            self.output.setPixmap(QPixmap.fromImage(qimage))

    def video_pred(self):
        ret, frame = self.video.read()
        if not ret:
            self.timer.stop()
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.input.setPixmap(QPixmap.fromImage(convert2QImage(frame)))
            results = self.model(frame)
            image = results.render()[0]
            self.output.setPixmap(QPixmap.fromImage(convert2QImage(image)))

    def open_video(self):
        print("点击了检测视频！")
        file_path = QFileDialog.getOpenFileName(self, dir="./datasets", filter="*.mp4")
        if file_path[0]:
            file_path = file_path[0]
            self.video = cv2.VideoCapture(file_path)
            self.timer.start()

    def quit(self):
        print("点击了退出程序！")
        sys.exit(0)

    def bind_slots(self):
        self.detect_image.clicked.connect(self.open_image)
        self.detect_video.clicked.connect(self.open_video)
        self.exit.clicked.connect(self.quit)
        self.timer.timeout.connect(self.video_pred)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec()