from ultralytics import YOLO

model = YOLO('yolov8s.pt')

model.train(data = './school_dataset.yaml', epoches = 100)