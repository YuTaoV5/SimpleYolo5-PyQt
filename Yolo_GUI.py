import sys
import serial
import serial.tools.list_ports
import time
from typing import List
import random

import json,xlwt
from PyQt5 import QtWidgets
from gui import Ui_Form
import cv2
import numpy as np
import math
import json
from multiprocessing import Pool
import shutil
import PyQt5.QtCore
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import threading
import argparse
import os
import sys
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import os.path as osp

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
class MainWindows(QtWidgets.QWidget, Ui_Form):

    def __init__(self):
        super(MainWindows, self).__init__()
        self.setupUi(self)
        self.init()
        self.setWindowTitle("Yolo上位机")
        self.ser = serial.Serial()
        self.port_check()
        self.setWindowIcon(QIcon("images/UI/logo.jpg"))

        # 接收数据和发送数据数目置零
        self.data_num_received = 0
        self.data_num_sended = 0
        self.rec_lcdNumber.display(self.data_num_received)
        self.send_lcdNumber.display(self.data_num_sended)

        # 图片读取进程
        self.output_size = 480
        self.img2predict = ""
        self.device = 'cpu'
        # # 初始化视频读取线程
        self.vid_source = '0'  # 初始设置为摄像头
        self.stopEvent = threading.Event()
        self.webcam = True
        self.stopEvent.clear()
        self.pushButton_streaming.setEnabled(False)
        self.pushButton_loadmp4.setEnabled(False)
        self.label_video.setPixmap(QPixmap("images/UI/yolo.jpg"))

        self.label_video.setScaledContents(True)
        self.vid_source = '0'
        self.webcam = True
        #self.model = self.model_load(weights="runs/train/exp12/weights/best.pt",device=self.device)  # todo 指明模型加载的位置的设备
        #self.reset_vid()

    '''
    ***模型初始化***
    '''
    @torch.no_grad()
    def model_load(self, weights="",  # model.pt path(s)
                   device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                   half=False,  # use FP16 half-precision inference
                   dnn=False,  # use OpenCV DNN for ONNX inference
                   ):

        device = select_device(device)
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        # Half
        half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt:
            model.model.half() if half else model.model.float()
        print("模型加载完成!")
        return model
    def reset_vid(self):
        self.pushButton_streaming.setEnabled(True)
        self.pushButton_loadmp4.setEnabled(True)
        self.label_video.setPixmap(QPixmap("images/UI/yolo.jpg"))

        self.label_video.setScaledContents(True)
        self.vid_source = '0'
        self.webcam = True
    def init(self):
        # 串口检测按钮
        self.box_1.clicked.connect(self.port_check)
        # 串口信息显示
        self.box_2.currentTextChanged.connect(self.port_imf)

        # 打开串口按钮
        self.open_button.clicked.connect(self.port_open)

        # 关闭串口按钮
        self.close_button.clicked.connect(self.port_close)

        # 发送数据按钮
        self.send_button.clicked.connect(self.data_send)

        # 定时发送数据
        self.timer_send = QTimer()
        self.timer_send.timeout.connect(self.data_send)
        self.timer_send_cb.stateChanged.connect(self.data_send_timer)

        # 定时器接收数据
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.data_receive)

        # 清除发送窗口
        self.clear_button.clicked.connect(self.send_data_clear)

        # 清除接收窗口
        self.clear_button.clicked.connect(self.receive_data_clear)

        # 上传图片窗口
        self.pushButton_loadpic.clicked.connect(self.upload_img)

        # 检测图片窗口
        self.pushButton_scanpic.clicked.connect(self.detect_img)
        # 检测视频窗口
        self.pushButton_streaming.clicked.connect(self.open_cam)
        self.pushButton_loadmp4.clicked.connect(self.open_mp4)
        self.pushButton_stopscan.clicked.connect(self.close_vid)

        # 打开模型
        self.pushButton_model.clicked.connect(self.open_model)
        # 保存结果
        self.pushButton_result.clicked.connect(self.save_res)
        self.pushButton_xls.clicked.connect(self.save_xls)
    '''
        打开模型
    '''

    def open_model(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.pt')
        if fileName:
            self.model = self.model_load(weights=fileName,
                                         device=self.device)  # todo 指明模型加载的位置的设备
            self.reset_vid()
    '''
        保存结果
    '''

    def save_res(self):
        filepath, type = QFileDialog.getSaveFileName(self, "文件保存", "/",
                                                     'txt(*.txt)')  # 前面是地址，后面是文件类型,得到输入地址的文件名和地址txt(*.txt*.xls);;image(*.png)不同类别
        file = open(filepath, 'w')
        print(filepath)
        with open("P:\\Item\\Yolov5\\yolov5-mask-42-master\\temp.txt", "r") as f:
            txt = f.read()

        file.write(txt)

    def save_xls(self):

        with open("P:\\Item\\Yolov5\\yolov5-mask-42-master\\temp.txt", "r") as f:
            xls = f.readlines()
        book = xlwt.Workbook(encoding='utf-8', style_compression=0)
        sheet = book.add_sheet('number', cell_overwrite_ok=True)
        sheet.write(0, 0, xls)
        filepath, type = QFileDialog.getSaveFileName(self, '文件保存', '/', 'xls(*.xls)')
        print(filepath)
        book.save(filepath)


    '''
    
    ***上传图片***
    '''
    def upload_img(self):
        # 选择录像文件进行读取
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
        if fileName:
            suffix = fileName.split(".")[-1]
            save_path = osp.join("images/tmp", "tmp_upload." + suffix)
            shutil.copy(fileName, save_path)
            # 应该调整一下图片的大小，然后统一防在一起
            im0 = cv2.imread(save_path)
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("images/tmp/upload_show_result.jpg", im0)
            # self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))
            self.img2predict = fileName
            self.left_img.setPixmap(QPixmap("images/tmp/upload_show_result.jpg"))
            self.left_img.setScaledContents(True)

            # todo 上传图片之后右侧的图片重置，
            self.right_img.setPixmap(QPixmap("images/UI/logo.jpg"))
            self.right_img.setScaledContents(True)
    '''
    ***检测图片***
    '''
    def detect_img(self):
        # 清空文本
        open("P:\\Item\\Yolov5\\yolov5-mask-42-master\\temp.txt", 'w').close()
        model = self.model
        output_size = self.output_size
        source = self.img2predict  # file/dir/URL/glob, 0 for webcam
        imgsz = 640  # inference size (pixels)
        conf_thres = 0.25  # confidence threshold
        iou_thres = 0.45  # NMS IOU threshold
        max_det = 1000  # maximum detections per image
        device = self.device  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img = False  # show results
        save_txt = False  # save results to *.txt
        save_conf = False  # save confidences in --save-txt labels
        save_crop = False  # save cropped prediction boxes
        nosave = False  # do not save images/videos
        classes = None  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms = False  # class-agnostic NMS
        augment = False  # ugmented inference
        visualize = False  # visualize features
        line_thickness = 3  # bounding box thickness (pixels)
        hide_labels = False  # hide labels
        hide_conf = False  # hide confidences
        half = False  # use FP16 half-precision inference
        dnn = False  # use OpenCV DNN for ONNX inference
        print(source)
        if source == "":
            QMessageBox.warning(self, "请上传", "请先上传图片再进行检测")
        else:
            source = str(source)
            device = select_device(self.device)
            webcam = False
            stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
            imgsz = check_img_size(imgsz, s=stride)  # check image size
            save_img = not nosave and not source.endswith('.txt')  # save inference images
            # Dataloader
            if webcam:
                view_img = check_imshow()
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
                bs = len(dataset)  # batch_size
            else:
                dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
                bs = 1  # batch_size
            vid_path, vid_writer = [None] * bs, [None] * bs
            # Run inference
            if pt and device.type != 'cpu':
                model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
            dt, seen = [0.0, 0.0, 0.0], 0
            for path, im, im0s, vid_cap, s in dataset:
                t1 = time_sync()
                im = torch.from_numpy(im).to(device)
                im = im.half() if half else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                t2 = time_sync()
                dt[0] += t2 - t1
                # Inference
                # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)
                t3 = time_sync()
                dt[1] += t3 - t2
                # NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                dt[2] += time_sync() - t3
                # Second-stage classifier (optional)
                # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
                # Process predictions
                for i, det in enumerate(pred):  # per image
                    seen += 1
                    if webcam:  # batch_size >= 1
                        p, im0, frame = path[i], im0s[i].copy(), dataset.count
                        s += f'{i}: '
                    else:
                        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                    p = Path(p)  # to Path
                    s += '%gx%g ' % im.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    imc = im0.copy() if save_crop else im0  # for save_crop
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                    -1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                # with open(txt_path + '.txt', 'a') as f:
                                #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            if save_img or save_crop or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                str_box = annotator.box_label(xyxy, label, color=colors(c, True))

                                t = str(xyxy)+str(label)
                                with open("P:\\Item\\Yolov5\\yolov5-mask-42-master\\temp.txt", "a") as f:
                                    f.write("第"+ str(seen) +"帧 : "+str_box + "\n")
                                self.textBrowser_pic.setText(t)
                                #if save_crop:
                                #     save_one_box(xyxy, imc, file="P:/Item/Yolov5/yolov5-mask-42-master/images/box" / names[c] / f'{p.stem}.jpg',
                                #                 BGR=True)
                    # Print time (inference-only)
                    LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
                    # Stream results
                    im0 = annotator.result()
                    # if view_img:
                    #     cv2.imshow(str(p), im0)
                    #     cv2.waitKey(1)  # 1 millisecond
                    # Save results (image with detections)
                    resize_scale = output_size / im0.shape[0]
                    im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
                    cv2.imwrite("images/tmp/single_result.jpg", im0)
                    # 目前的情况来看，应该只是ubuntu下会出问题，但是在windows下是完整的，所以继续

                    self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))
                    self.right_img.setScaledContents(True)
    # 视频检测，逻辑基本一致，有两个功能，分别是检测摄像头的功能和检测视频文件的功能，先做检测摄像头的功能。

    '''
    ### 界面关闭事件 ### 
    '''
    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     '退出',
                                     "Are you sure?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()

    '''
    ### 视频关闭事件 ### 
    '''

    def open_cam(self):
        self.pushButton_streaming.setEnabled(False)
        self.pushButton_loadmp4.setEnabled(False)
        self.pushButton_stopscan.setEnabled(True)
        self.vid_source = '0'
        self.webcam = True
        # 把按钮给他重置了
        # print("GOGOGO")
        th = threading.Thread(target=self.detect_vid)
        th.start()

    '''
    ### 开启视频文件检测事件 ### 
    '''

    def open_mp4(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.mp4 *.avi')
        if fileName:
            self.pushButton_streaming.setEnabled(False)
            self.pushButton_loadmp4.setEnabled(False)
            # self.pushButton_stopscan.setEnabled(True)
            self.vid_source = fileName
            self.webcam = False
            th = threading.Thread(target=self.detect_vid)
            th.start()

    '''
    ### 视频开启事件 ### 
    '''

    # 视频和摄像头的主函数是一样的，不过是传入的source不同罢了
    def detect_vid(self):
        # 清空文本
        open("P:\\Item\\Yolov5\\yolov5-mask-42-master\\temp.txt", 'w').close()
        # pass
        model = self.model
        output_size = self.output_size
        # source = self.img2predict  # file/dir/URL/glob, 0 for webcam
        imgsz = 640  # inference size (pixels)
        conf_thres = 0.25  # confidence threshold
        iou_thres = 0.45  # NMS IOU threshold
        max_det = 1000  # maximum detections per image
        # device = self.device  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img = False  # show results
        save_txt = False  # save results to *.txt
        save_conf = False  # save confidences in --save-txt labels
        save_crop = False  # save cropped prediction boxes
        nosave = False  # do not save images/videos
        classes = None  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms = False  # class-agnostic NMS
        augment = False  # ugmented inference
        visualize = False  # visualize features
        line_thickness = 3  # bounding box thickness (pixels)
        hide_labels = False  # hide labels
        hide_conf = False  # hide confidences
        half = False  # use FP16 half-precision inference
        dnn = False  # use OpenCV DNN for ONNX inference
        source = str(self.vid_source)
        webcam = self.webcam
        device = select_device(self.device)
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        # Dataloader
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs
        # Run inference
        if pt and device.type != 'cpu':
            model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0
        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1
            # Inference
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2
            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3
            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # save_path = str(save_dir / p.name)  # im.jpg
                # txt_path = str(save_dir / 'labels' / p.stem) + (
                #     '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                cv2.imwrite("images/tmp/single_org_vid.jpg", im0)
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    print(len(det))
                    print(det[0][4])
                    # 实现在所有标注框中只标注置信度最大的det_max
                    MaxConf = []
                    len_det = len(det)
                    for i in range(len_det):
                        # print(i)
                        MaxConf.append(det[i][4])
                        Max = max(MaxConf)
                        # 最大值
                        MaxI = MaxConf.index(Max)
                    #det_max = det[[MaxI]]
                    det = det[[MaxI]]
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                -1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            # with open(txt_path + '.txt', 'a') as f:
                            #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            str_box = annotator.box_label(xyxy, label, color=colors(c, True))
                            t = str(xyxy)+str(label)
                            with open("P:\\Item\\Yolov5\\yolov5-mask-42-master\\temp.txt", "a") as f:
                                f.write("第"+ str(seen) +"帧 : "+str_box + "\n")

                                self.textBrowser_video.setText(str_box)

                            # if save_crop:
                            #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg',
                            #                  BGR=True)
                #self.label_video.setText(str_box)
                # Print time (inference-only)
                LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
                # Stream results
                # Save results (image with detections)

                im0 = annotator.result()
                frame = im0
                resize_scale = output_size / frame.shape[0]
                frame_resized = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
                cv2.imwrite("images/tmp/single_result_vid.jpg", frame_resized)


                #self.label_video.setPixmap(QPixmap("test.jpg"))
                self.label_video.setPixmap(QPixmap("images/tmp/single_result_vid.jpg"))
                self.label_video.setScaledContents(True)

                # self.label_video
                # if view_img:
                # cv2.imshow(str(p), im0)
                # self.label_video.setPixmap(QPixmap("images/tmp/single_result_vid.jpg"))
                # cv2.waitKey(1)  # 1 millisecond

            if cv2.waitKey(25) & self.stopEvent.is_set() == True:
                self.stopEvent.clear()
                self.pushButton_streaming.setEnabled(True)
                self.pushButton_loadmp4.setEnabled(True)
                self.reset_vid()
                break
        # self.reset_vid()


    '''
    ### 视频重置事件 ### 
    '''

    def close_vid(self):
        self.stopEvent.set()
        self.reset_vid()


    # 串口检测
    def port_check(self):
        # 检测所有存在的串口，将信息存储在字典中
        self.Com_Dict = {}
        port_list = list(serial.tools.list_ports.comports())
        self.box_2.clear()
        for port in port_list:
            self.Com_Dict["%s" % port[0]] = "%s" % port[1]
            self.box_2.addItem(port[0])
        if len(self.Com_Dict) == 0:
            self.state_label.setText(" 无串口")

    # 串口信息
    def port_imf(self):
        # 显示选定的串口的详细信息
        imf_s = self.box_2.currentText()
        if imf_s != "":
            self.state_label.setText(self.Com_Dict[self.box_2.currentText()])

    # 打开串口
    def port_open(self):
        self.ser.port = self.box_2.currentText()
        self.ser.baudrate = int(self.box_3.currentText())
        self.ser.bytesize = int(self.box_4.currentText())
        self.ser.stopbits = int(self.box_6.currentText())
        self.ser.parity = self.box_5.currentText()

        try:
            self.ser.open()
        except:
            QMessageBox.critical(self, "Port Error", "此串口不能被打开！")
            return None

        # 打开串口接收定时器，周期为2ms
        self.timer.start(2)

        if self.ser.isOpen():
            self.open_button.setEnabled(False)
            self.close_button.setEnabled(True)
            self.formGroupBox.setTitle("串口状态（已开启）")

    # 关闭串口
    def port_close(self):
        self.timer.stop()
        self.timer_send.stop()
        try:
            self.ser.close()
        except:
            pass
        self.open_button.setEnabled(True)
        self.close_button.setEnabled(False)
        self.lineEdit_3.setEnabled(True)
        # 接收数据和发送数据数目置零
        self.data_num_received = 0
        #self.lineEdit.setText(str(self.data_num_received))
        self.data_num_sended = 0
        #self.lineEdit_2.setText(str(self.data_num_sended))
        self.rec_lcdNumber.display(self.data_num_received)
        self.send_lcdNumber.display(self.data_num_sended)
        self.formGroupBox.setTitle("串口状态（已关闭）")

    # 发送数据
    def data_send(self):
        if self.ser.isOpen():
            input_s = self.send_text.toPlainText()
            if input_s != "":
                # 非空字符串
                if self.hex_send.isChecked():
                    # hex发送
                    input_s = input_s.strip()
                    send_list = []
                    while input_s != '':
                        try:
                            num = int(input_s[0:2], 16)
                        except ValueError:
                            QMessageBox.critical(self, 'wrong data', '请输入十六进制数据，以空格分开!')
                            return None
                        input_s = input_s[2:].strip()
                        send_list.append(num)
                    input_s = bytes(send_list)
                else:
                    # ascii发送
                    input_s = (input_s + '\r\n').encode('utf-8')

                num = self.ser.write(input_s)
                self.data_num_sended += num
                #self.lineEdit_2.setText(str(self.data_num_sended))
                self.send_lcdNumber.display(self.data_num_sended)
        else:
            pass

    # 接收数据
    def data_receive(self):
        try:
            num = self.ser.inWaiting()
        except:
            self.port_close()
            return None
        if num > 0:
            data = self.ser.read(num)
            num = len(data)
            # hex显示
            if self.hex_receive.checkState():
                out_s = ''
                for i in range(0, len(data)):
                    out_s = out_s + '{:02X}'.format(data[i]) + ' '
                self.receive_text.insertPlainText(out_s)
            else:
                # 串口接收到的字符串为b'123',要转化成unicode字符串才能输出到窗口中去
                self.receive_text.insertPlainText(data.decode('iso-8859-1'))

            # 统计接收字符的数量
            self.data_num_received += num
            #self.lineEdit.setText(str(self.data_num_received))
            self.rec_lcdNumber.display(self.data_num_received)
            # 获取到text光标
            textCursor = self.receive_text.textCursor()
            # 滚动到底部
            textCursor.movePosition(textCursor.End)
            # 设置光标到text中去
            self.receive_text.setTextCursor(textCursor)
        else:
            pass

    # 定时发送数据
    def data_send_timer(self):
        if self.timer_send_cb.isChecked():
            self.timer_send.start(int(self.lineEdit_3.text()))
            self.lineEdit_3.setEnabled(False)
        else:
            self.timer_send.stop()
            self.lineEdit_3.setEnabled(True)

    # 清除显示
    def send_data_clear(self):
        self.send_text.setText("")

    def receive_data_clear(self):
        self.receive_text.setText("")


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    myshow = MainWindows()
    myshow.show()
    sys.exit(app.exec_())
