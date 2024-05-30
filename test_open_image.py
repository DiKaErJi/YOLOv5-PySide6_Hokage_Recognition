# 此文件用于测试为什么检测后图片可以显示但是原图片显示不出来！！！！为什么！！！！！

import os
import sys
from PIL import Image
from PySide6.QtWidgets import QApplication, QLabel, QFileDialog
from PySide6.QtGui import QPixmap, QImage
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = r'D:\software\Anaconda\envs\yolov5\Lib\site-packages\PySide6\plugins\platforms'# 设置Qt平台插件路径的环境变量


def convert_jpeg_to_png(jpeg_path):
    """Convert JPEG image to PNG format."""
    with Image.open(jpeg_path) as img:
        png_path = jpeg_path.replace('.jpg', '.png').replace('.jpeg', '.png')
        img.save(png_path)
    return png_path


def load_image(file_path):
    """Load an image, converting to PNG if necessary."""
    if file_path.lower().endswith(('.jpg', '.jpeg')):
        file_path = convert_jpeg_to_png(file_path)

    pixmap = QPixmap(file_path)
    if pixmap.isNull():
        print(f"Failed to load image from {file_path}")
    return pixmap


def open_image():
    file_path = QFileDialog.getOpenFileName(
        None, "Open Image", "D:\Temporary", "Images (*.jpg *.jpeg *.png)"
    )[0]

    if file_path:
        pixmap = load_image(file_path)
        if not pixmap.isNull():
            label.setPixmap(pixmap)
            label.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    label = QLabel()

    open_image()

    sys.exit(app.exec())