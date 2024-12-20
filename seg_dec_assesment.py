import cv2
from ultralytics import YOLO

# 모델 로드
detection_model = YOLO('/Users/mac/Desktop/github/thesis/yolov8m.pt')
segmentation_model = YOLO('/Users/mac/Desktop/github/thesis/YOLO_SEG_NCNN/weights/yolov8/both/1004/weights/best.pt')

# 이미지 경로 설정
image_path = '/Users/mac/Desktop/github/thesis/YOLO_DEC_NCNN/shcool-dataset/train/images/FILE240509-093910.AVI_image_8.jpg'

# Detection 결과
detection_results = detection_model(image_path)
detection_plot = detection_results[0].plot()

# Segmentation 결과
segmentation_results = segmentation_model(image_path)
segmentation_plot = segmentation_results[0].plot()

# 결과 출력
cv2.imshow("Detection Result", detection_plot)
cv2.imshow("Segmentation Result", segmentation_plot)

cv2.waitKey(0)
cv2.destroyAllWindows()
