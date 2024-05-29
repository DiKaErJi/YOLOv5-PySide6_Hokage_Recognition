# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch

# 定义当前文件路径
FILE = Path(__file__).resolve()
# 定义 YOLOv5 根目录
ROOT = FILE.parents[0]  # YOLOv5 root directory
# 如果 YOLOv5 根目录不在系统路径中，则将其添加到系统路径中
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# 将 YOLOv5 根目录相对于当前工作目录的路径设置为 ROOT
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# 导入必要的模块
from models.common import DetectMultiBackend  # 用于加载模型
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams  # 用于加载数据
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)  # 用于通用工具
from utils.plots import Annotator, colors, save_one_box  # 用于绘制结果
from utils.torch_utils import select_device, smart_inference_mode  # 用于选择设备和智能推理模式


# 使用智能推理模式装饰器，该装饰器会根据模型类型自动选择推理模式
@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # 模型路径或 Triton URL
        source=ROOT / 'data/images',  # 文件/目录/URL/glob/屏幕/0(网络摄像头)
        data=ROOT / 'data/coco128.yaml',  # 数据集 yaml 路径
        imgsz=(640, 640),  # 推理尺寸 (高度, 宽度)
        conf_thres=0.25,  # 置信度阈值
        iou_thres=0.45,  # NMS IoU 阈值
        max_det=1000,  # 每张图像的最大检测数
        device='',  # CUDA 设备，例如 0 或 0,1,2,3 或 cpu
        view_img=False,  # 显示结果
        save_txt=False,  # 将结果保存到 *.txt
        save_conf=False,  # 在 --save-txt 标签中保存置信度
        save_crop=False,  # 保存裁剪后的预测框
        nosave=False,  # 不保存图像/视频
        classes=None,  # 按类别过滤：--class 0，或 --class 0 2 3
        agnostic_nms=False,  # 类无关 NMS
        augment=False,  # 增强的推理
        visualize=False,  # 可视化特征
        update=False,  # 更新所有模型
        project=ROOT / 'runs/detect',  # 将结果保存到 project/name
        name='exp',  # 将结果保存到 project/name
        exist_ok=False,  # 现有 project/name 允许，不递增
        line_thickness=3,  # 边框厚度 (像素)
        hide_labels=False,  # 隐藏标签
        hide_conf=False,  # 隐藏置信度
        half=False,  # 使用 FP16 半精度推理
        dnn=False,  # 使用 OpenCV DNN 进行 ONNX 推理
        vid_stride=1,  # 视频帧率步长
):
    source = str(source)
    # 如果不保存或源文件是 .txt 文件，则不保存推理图像
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    # 检查源文件是否为图像或视频文件
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    # 检查源文件是否为 URL
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    # 检查源文件是否为网络摄像头
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    # 检查源文件是否为屏幕截图
    screenshot = source.lower().startswith('screen')
    # 如果源文件是 URL 并且是文件，则下载文件
    if is_url and is_file:
        source = check_file(source)  # download

    # 目录
    # 创建保存结果的目录，如果目录已存在，则递增目录名称
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # 创建保存标签的目录，如果保存标签，则创建目录
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # 加载模型
    # 选择设备
    device = select_device(device)
    # 加载模型
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    # 获取模型步长、类别名称和是否为 PyTorch 模型
    stride, names, pt = model.stride, model.names, model.pt
    # 检查图像尺寸是否与模型步长兼容
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # 数据加载器
    # 设置批次大小
    bs = 1  # batch_size
    # 如果是网络摄像头，则设置显示结果，并加载网络摄像头数据
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        # 设置批次大小为网络摄像头数量
        bs = len(dataset)
    # 如果是屏幕截图，则加载屏幕截图数据
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    # 否则，加载图像或视频数据
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    # 初始化视频路径和视频写入器
    vid_path, vid_writer = [None] * bs, [None] * bs

    # 运行推理
    # 对模型进行预热
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    # 初始化计数器、窗口列表和计时器
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    # 遍历数据集
    for path, im, im0s, vid_cap, s in dataset:
        # 使用计时器记录预处理时间
        with dt[0]:
            # 将图像转换为 PyTorch 张量并移动到设备
            im = torch.from_numpy(im).to(model.device)
            # 如果使用 FP16，则将图像转换为 FP16，否则转换为 FP32
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            # 将图像像素值归一化到 0.0 - 1.0 之间
            im /= 255  # 0 - 255 to 0.0 - 1.0
            # 如果图像的维度为 3，则扩展为批次维度
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # 推理
        # 使用计时器记录推理时间
        with dt[1]:
            # 如果需要可视化特征，则创建可视化目录
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            # 进行推理
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        # 使用计时器记录 NMS 时间
        with dt[2]:
            # 进行非极大值抑制
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # 二阶段分类器 (可选)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # 处理预测结果
        # 遍历每张图像的预测结果
        for i, det in enumerate(pred):  # per image
            # 计数器加 1
            seen += 1
            # 如果是网络摄像头，则获取图像路径、原始图像和帧数
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            # 否则，获取图像路径、原始图像和帧数
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            # 将图像路径转换为 Path 对象
            p = Path(p)  # to Path
            # 获取保存图像的路径
            save_path = str(save_dir / p.name)  # im.jpg
            # 获取保存标签的路径
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            # 打印图像尺寸
            s += '%gx%g ' % im.shape[2:]  # print string
            # 获取归一化增益
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # 如果需要保存裁剪后的预测框，则复制原始图像，否则使用原始图像
            imc = im0.copy() if save_crop else im0  # for save_crop
            # 创建注释器对象
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            # 如果有检测结果
            if len(det):
                # 将预测框从推理尺寸缩放至原始图像尺寸
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # 打印结果
                # 遍历所有类别
                for c in det[:, 5].unique():
                    # 计算每个类别的检测数
                    n = (det[:, 5] == c).sum()  # detections per class
                    # 将检测结果添加到字符串中
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # 写入结果
                # 反向遍历检测结果
                for *xyxy, conf, cls in reversed(det):
                    # 如果需要保存标签
                    if save_txt:  # Write to file
                        # 将预测框转换为 xywh 格式并归一化
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # 创建标签格式
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        # 将标签写入文件
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # 如果需要保存图像、保存裁剪后的预测框或显示结果
                    if save_img or save_crop or view_img:  # Add bbox to image
                        # 将类别转换为整数
                        c = int(cls)  # integer class
                        # 创建标签
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        # 在图像上绘制预测框和标签
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    # 如果需要保存裁剪后的预测框
                    if save_crop:
                        # 保存裁剪后的预测框
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # 流式结果
            # 获取注释后的图像
            im0 = annotator.result()
            # 如果需要显示结果
            if view_img:
                # 如果是 Linux 系统，并且图像路径不在窗口列表中
                if platform.system() == 'Linux' and p not in windows:
                    # 将图像路径添加到窗口列表中
                    windows.append(p)
                    # 创建窗口
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    # 调整窗口大小
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                # 显示图像
                cv2.imshow(str(p), im0)
                # 等待 1 毫秒
                cv2.waitKey(1)  # 1 millisecond

            # 保存结果 (带有检测结果的图像)
            # 如果需要保存图像
            if save_img:
                # 如果是图像模式
                if dataset.mode == 'image':
                    # 保存图像
                    cv2.imwrite(save_path, im0)
                # 否则，如果是视频或流模式
                else:  # 'video' or 'stream'
                    # 如果视频路径与保存路径不同，则创建新的视频写入器
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        # 如果视频写入器已存在，则释放之前的视频写入器
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        # 如果是视频
                        if vid_cap:  # video
                            # 获取视频帧率、宽度和高度
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        # 否则，如果是流
                        else:  # stream
                            # 设置帧率、宽度和高度
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        # 设置保存路径为 .mp4 格式
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        # 创建视频写入器
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    # 将图像写入视频
                    vid_writer[i].write(im0)

        # 打印时间 (仅推理时间)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # 打印结果
    # 计算每张图像的预处理时间、推理时间和 NMS 时间
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    # 打印速度信息
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    # 如果需要保存标签或图像
    if save_txt or save_img:
        # 打印保存标签的信息
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # 打印保存结果的信息
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    # 如果需要更新模型
    if update:
        # 移除优化器
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


# 解析命令行参数
def parse_opt():
    parser = argparse.ArgumentParser()
    # 添加模型路径参数
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    # 添加源文件参数
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    # 添加数据集 yaml 路径参数
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    # 添加推理尺寸参数
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    # 添加置信度阈值参数
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    # 添加 NMS IoU 阈值参数
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    # 添加每张图像的最大检测数参数
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    # 添加 CUDA 设备参数
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # 添加显示结果参数
    parser.add_argument('--view-img', action='store_true', help='show results')
    # 添加保存标签参数
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # 添加保存置信度参数
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # 添加保存裁剪后的预测框参数
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    # 添加不保存图像/视频参数
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # 添加按类别过滤参数
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    # 添加类无关 NMS 参数
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # 添加增强推理参数
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    # 添加可视化特征参数
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    # 添加更新所有模型参数
    parser.add_argument('--update', action='store_true', help='update all models')
    # 添加保存结果目录参数
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    # 添加保存结果名称参数
    parser.add_argument('--name', default='exp', help='save results to project/name')
    # 添加现有目录允许参数
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # 添加边框厚度参数
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    # 添加隐藏标签参数
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    # 添加隐藏置信度参数
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    # 添加使用 FP16 半精度推理参数
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    # 添加使用 OpenCV DNN 进行 ONNX 推理参数
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    # 添加视频帧率步长参数
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    # 解析参数
    opt = parser.parse_args()
    # 如果推理尺寸只有一个值，则将其扩展为两个值
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    # 打印参数
    print_args(vars(opt))
    # 返回参数
    return opt


# 主函数
def main(opt):
    # 检查依赖项
    check_requirements(exclude=('tensorboard', 'thop'))
    # 运行推理
    run(**vars(opt))


# 如果是主程序，则运行主函数
if __name__ == "__main__":
    # 解析命令行参数
    opt = parse_opt()
    # 运行主函数
    main(opt)