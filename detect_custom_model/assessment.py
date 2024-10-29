import cv2
from ultralytics import YOLO

# model = YOLO('yolov8s.pt')

model = YOLO('C:\\Users\\shims\\Desktop\\github\\thesis\\runs\\detect\\train4\\weights\\best.pt')
results = model('C:\\Users\\shims\\Desktop\\github\\thesis\\yolo_detect_model\\shcool-dataset\\train\\images\\FILE240508-095102.AVI_image_22.jpg')

plots = results[0].plot()
cv2.imshow("plot", plots)
cv2.waitKey(0)
cv2.destroyAllWindows()
