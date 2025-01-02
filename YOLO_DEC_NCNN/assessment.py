import cv2
from ultralytics import YOLO

# model = YOLO('yolov8s.pt')

model = YOLO('C://Users//shims//Desktop//github//thesis//yolov8m.pt')

results = model('C://Users//shims//Desktop//github//thesis//FILE240513-083936.AVI_image_15.jpg')

plots = results[0].plot()
cv2.imshow("plot", plots)
cv2.waitKey(0)
cv2.destroyAllWindows()
