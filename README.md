<a href="https://apps.apple.com/app/id1452689527" target="_blank">
<img src="https://user-images.githubusercontent.com/26833433/82944393-f7644d80-9f4f-11ea-8b87-1a5b04f555f1.jpg" width="1000"></a>
&nbsp

![CI CPU testing](https://github.com/ultralytics/yolov5/workflows/CI%20CPU%20testing/badge.svg)

This repository represents Ultralytics open-source research into future object detection methods, and incorporates our lessons learned and best practices evolved over training thousands of models on custom client datasets with our previous YOLO repository https://github.com/ultralytics/yolov3. **All code and models are under active development, and are subject to modification or deletion without notice.** Use at your own risk.

And This repository made for main software of Mask Detection and Tempature Scanning Auto-Access-Control-Gate.

----------------------------------

Environment

Jetson Nano with Jetpack v4.5.1 
Anaconda
YOLOv5 v3.1 
Python v3.6
--------------------------------

Modules For Mask Detection

Cython==0.29.23
future==0.18.2
matplotlib==3.2.2
numpy==1.19.3
opencv-python==4.5.1.48
pandas==1.1.5
Pillow==8.2.0
PyYAML==5.4.1
scikit-build==0.11.1
scipy==1.5.4
seaborn==0.11.1
tensorboard==2.5.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.0
torch @ torch-1.8.0-cp36-cp36m-linux_aarch64.whl
torchvision==0.9.0a0+01dfa8e
tqdm==4.60.0
-----------------------------------

Modules For Tempature Scanning

pyserial==3.5
serial==0.0.97
websockets==9.0.2
-------------------------------------





## Inference

detect.py runs inference on a variety of sources, downloading models automatically from the [latest YOLOv5 release](https://github.com/ultralytics/yolov5/releases) and saving results to `inference/output`.
```bash
$ python detect.py --source 0  # webcam
                            file.jpg  # image 
                            file.mp4  # video
                            path/  # directory
                            path/*.jpg  # glob
                            rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa  # rtsp stream
                            rtmp://192.168.1.105/live/test  # rtmp stream
                            http://112.50.243.8/PLTV/88888888/224/3221225900/1.m3u8  # http stream
```

To run inference on example images in `inference/images`:
```bash
$ python detect.py --source inference/images --weights yolov5s.pt --conf 0.25

Namespace(agnostic_nms=False, augment=False, classes=None, conf_thres=0.25, device='', img_size=640, iou_thres=0.45, output='inference/output', save_conf=False, save_txt=False, source='inference/images', update=False, view_img=False, weights='yolov5s.pt')
Using CUDA device0 _CudaDeviceProperties(name='Tesla V100-SXM2-16GB', total_memory=16160MB)

Downloading https://github.com/ultralytics/yolov5/releases/download/v3.0/yolov5s.pt to yolov5s.pt... 100%|██████████████| 14.5M/14.5M [00:00<00:00, 21.3MB/s]

Fusing layers... 
Model Summary: 140 layers, 7.45958e+06 parameters, 0 gradients
image 1/2 yolov5/inference/images/bus.jpg: 640x480 4 persons, 1 buss, 1 skateboards, Done. (0.013s)
image 2/2 yolov5/inference/images/zidane.jpg: 384x640 2 persons, 2 ties, Done. (0.013s)
Results saved to yolov5/inference/output
Done. (0.124s)
```
<img src="https://user-images.githubusercontent.com/26833433/97107365-685a8d80-16c7-11eb-8c2e-83aac701d8b9.jpeg" width="500">  



## Training

Download [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh) and run command below. Training times for YOLOv5s/m/l/x are 2/4/6/8 days on a single V100 (multi-GPU times faster). Use the largest `--batch-size` your GPU allows (batch sizes shown for 16 GB devices).
```bash
$ python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
                                         yolov5m                                40
                                         yolov5l                                24
                                         yolov5x                                16
```
<img src="https://user-images.githubusercontent.com/26833433/90222759-949d8800-ddc1-11ea-9fa1-1c97eed2b963.png" width="900">


