from ultralytics import YOLO

model = YOLO('yolov8s.pt')

model.train(data='./school_dataset.yaml', epochs=10)
