## YOLO Segmentation
### prerequisite
- Ubuntu20.04
- RTX 3090
```bash
pip install -r requirements.txt
```

### datasets
BDD100K
- drive available
- lane line
```
.
├── images
│   ├── test
│   │   └── fe1f55fa-19ba3600.png
│   ├── train
│   │   └── fe1f2409-c16ea1ed.png
│   └── val
│       └── fe1b3799-a7863feb.png
└── labels
    ├── train
    │   └── fe1f2409-c16ea1ed.txt
    └── val
        └── fe1b3799-a7863feb.txt
```
### [yolo format](https://docs.ultralytics.com/datasets/segment/#supported-dataset-formats)
- drive available : `-l da`
- lane line : `-l ll`
```bash
python converter/converter.py -l da
```
```bash
python converter/converter.py -l ll
```
### train
```bash
python main.py
```
### tensorboard
```bash
tensorboard --logdir=./weights
```
### demo
- checkpoint : `-c yolov8/ll/1123/weights/best`
- data : `-d test`
```bash
python demo.py -d test -c yolov8/ll/1123/weights/best
```
```bash
python demo.py -d test -c yolov8/da/1123/weights/best
```
```bash
python demo.py -c yolov8/both/1004/weights/best -d test
```
### export
```bash
python export.py -c yolov8/da/1123/weights/best
```

### model converter
https://convertmodel.com/
