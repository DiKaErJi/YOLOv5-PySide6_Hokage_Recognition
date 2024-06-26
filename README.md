# 火影忍者角色识别器

这是一个使用YOLOv5模型训练的火影忍者角色识别器，同时附带了一个基于PySide6制作的简单GUI界面。该项目可以识别火影忍者动漫中的三个角色：带土、鸣人和我爱罗。

## 功能特点

- **目标检测与识别**：利用YOLOv5模型进行火影忍者角色的目标检测和识别。
- **支持角色**：目前支持识别带土、鸣人和我爱罗这三个角色。
- **用户界面**：提供了一个简单易用的PySide6 GUI界面，方便用户上传图片并查看识别结果。

## 环境配置

### 依赖项安装

确保你的环境中安装了以下依赖项：

- **Python 3.8**
- **PyTorch**：用于YOLOv5模型的训练和推理。
- **PySide6**：用于制作GUI界面。
- **其他依赖项**：参考YOLOv5项目的安装要求，如NumPy、OpenCV等。

### 模型下载与配置

1. **下载YOLOv5模型**：从YOLOv5的官方仓库或其他可信的来源下载预训练的YOLOv5模型权重文件，并解压到本地文件夹。
2. **数据集准备**：准备包含火影忍者角色（带土、鸣人、我爱罗）的图像数据集，并将其整理为YOLOv5所需的数据格式（如`.yaml`文件）。
3. **配置文件**：根据你的数据集路径和模型路径，修改配置文件以确保模型能够正确加载和使用。

## 使用说明

### 运行GUI界面

1. **启动命令**：在命令行中执行以下命令启动GUI界面：

   ```python
   python base_ui.py
   ```

2. **界面操作**：GUI界面会显示一个简单的文件上传按钮，你可以选择一张图片文件上传，然后等待识别结果显示在界面上。

### 命令行使用

如果你更倾向于在命令行中使用，可以按以下步骤操作：

1. **运行检测脚本**：使用命令行运行检测脚本，指定图片或视频文件的路径、模型权重和类别名称文件。

   ```python
   python detect.py --source path/to/image/or/video --weights path/to/your/weights --names path/to/your/names
   ```

   - `path/to/image/or/video`：要检测的图像或视频文件的路径。
   - `path/to/your/weights`：YOLOv5模型的权重文件路径。
   - `path/to/your/names`：包含角色类别名称的文件路径。

2. **查看结果**：脚本会输出检测结果或将结果保存到指定位置。

## 示例

![image](https://github.com/DiKaErJi/YOLOv5-PySide6_Hokage_Recognition/blob/master/readme_images.png)

## 致谢

- **YOLOv5开源项目**：感谢YOLOv5团队提供强大的目标检测模型和文档支持。
- **PySide6文档**：PySide6提供了强大的GUI开发能力，让制作GUI界面变得简单和高效。
