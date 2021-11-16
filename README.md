<a href="https://apps.apple.com/app/id1452689527" target="_blank">
<img src="https://user-images.githubusercontent.com/26833433/82944393-f7644d80-9f4f-11ea-8b87-1a5b04f555f1.jpg" width="1000"></a>
&nbsp

![CI CPU testing](https://github.com/ultralytics/yolov5/workflows/CI%20CPU%20testing/badge.svg)

This repository represents Ultralytics open-source research into future object detection methods, and incorporates our lessons learned and best practices evolved over training thousands of models on custom client datasets with our previous YOLO repository https://github.com/ultralytics/yolov3. **All code and models are under active development, and are subject to modification or deletion without notice.** Use at your own risk.

And This repository made for main software of Mask Detection and Tempature Scanning Auto-Access-Control-Gate.

----------------------------------

## Environment

- Jetson Nano with Jetpack v4.5.1 
- Anaconda
- YOLOv5 v3.1 
- Python v3.6.13
--------------------------------

## Modules For Mask Detection

- Cython==0.29.23
- future==0.18.2
- matplotlib==3.2.2
- numpy==1.19.3
- opencv-python==4.5.1.48
- pandas==1.1.5
- Pillow==8.2.0
- PyYAML==5.4.1
- scikit-build==0.11.1
- scipy==1.5.4
- seaborn==0.11.1
- tensorboard==2.5.0
- tensorboard-data-server==0.6.1
- tensorboard-plugin-wit==1.8.0
- torch @ torch-1.8.0-cp36-cp36m-linux_aarch64.whl
- torchvision==0.9.0a0+01dfa8e
- tqdm==4.60.0
-----------------------------------

## Modules For Tempature Scanning

- pyserial==3.5
- serial==0.0.97
- websockets==9.0.2
-------------------------------------

## How To Consist Environment Of YOLOv5 In Jetson Nano

- 젯슨 나노에 일반적인 pip (v21.1.1)으로 torch와 torchvision을 설치시 manylinux2014_aarch64.whl 기반으로 설치된다.
- 이 버전은 젯슨 나노의 운영체제에 호환되지 않는다. (정확히는 CUDA, 즉 Jetson Nano의 GPU를 사용할 수 없다.)
- torch 설치 후 import torch를 인터프리터 하거나 컴파일 할 경우 Illegal instruction (core dumped) 라는 오류와 함께 프로세스가 강제 종료되는 오류가 있다.
- 이 문제는 파이썬 3.6.13에 기본적으로 포함된 numpy의 버전이 1.19.5 이상이며 운영체제에 호환되지 않기 때문에 발생하는 문제이므로, numpy==1.19.4를 설치해야한다.

- 해당 문제를 해결하기 위해 젯슨 나노의 운영체제에 맞는 torch와 torchvision을 설치해야한다.
- 또한 해당 torch와 torchvision 버전에 호환되는 YOLOv5 및 하위 모듈을 찾아 설치해야한다.
- 다음 과정을 주의깊게 따라하자.

- 1 젯슨 나노에 제트팩 v4.5.1를 설치한다.

- 2 Anaconda를 설치해 Python v3.6.13 기반 가상환경을 구성한다.

- 3 pip install pip==21.1.1 를 설치한다.

- 4 pip install numpy==1.19.4 를 설치한다.

- 5 YOLOv5 환경을 구성할 directory를 생성한다.
- mkdir your_own_dir_name
- cd dir (home/dir)

- 6 torch-1.8.0-cp36-cp36m-linux_aarch64.whl 를 설치한다.
- wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl
- cp p57jwntv436lfrd78inwl7iml6p13fzh.whl torch-1.8.0-cp36-cp36m-linux_aarch64.whl
- pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl
- python 실행 후 import torch를 했을 때 Illegal instruction (core dumped) 오류가 발생하지 않았다면 성공.
- 이 후 torch.cuda.is_available() 을 했을 때 True가 뜬다면 정상이고, False가 뜰 경우 torch 설치 시 torch-1.8.0-cp36-cp36m-linux_aarch64.whl 를 맞게 설치했는지 점검한다.

- 7 torchvision 0.9.0a0+01dfa8e를 설치한다.
- git clone https://github.com/pytorch/vision torchvision -b v0.9.0
- cd torchvision (home/dir/torchvision)
- python3 setup.py install (오래 걸림 약 30분)
- 이 부분에서 Illegal instruction (core dumped) 오류가 발생했다면 numpy 버전이 1.19.4인지 확인하자.

- 8 aarch64에서 호환되는 YOLOv5 v3.1을 설치한다.
- cd dir
- git clone https://github.com/ultralytics/yolov5 -b v3.1
- cd yolov5 (home/dir/yolov5)
- git checkout v3.1

- 9 이제 YOLOv5의 환경 설정을 해줄 차례이다.
- sudo nano requirements.txt 를 통해 나노 편집기에서 requirements.text에 access 한다.
- numpy>=1.18.5 삭제
- torch>=1.6.0 삭제
- torchvision>=0.7.0 삭제
- opencv-python>=4.1.2 에서 opencv-python==4.5.1.48 수정
- pandas==1.1.5 추가
- extras의 # thop에서 #를 삭제 (train시에 thop 모듈을 사용해야하므로 주석처리를 해제한다.)

- 10 pip install -r requirements.txt를 통해 하위 모듈을 설치한다. (오래 걸림 약 10분)

- 11 yolov5 디렉토리에서 python detect.py를 통해 기본 예제를 실행할 수 있다.


- detect 사용법 / train 방법 예정



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


