# SSA-SSAM
The project of YOLOv5s with SSA/SSAM 

Based on 
https://github.com/ultralytics/yolov5/releases/tag/v4.0

train: 

python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 48

python train.py --data coco.yaml --cfg yolov5saa.yaml --weights '' --batch-size 48

python train.py --data coco.yaml --cfg yolov5saa.yaml --weights '' --batch-size 48

python train.py --data coco.yaml --cfg yolov5seca.yaml  --weights '' --batch-size 48


test:  

python test.py --data coco.yaml --img 640 --conf 0.001 --iou 0.65

using pycocotools

The SSA module can be found in models/SSAmodule_example.py

