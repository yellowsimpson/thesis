import argparse
import os
from datetime import datetime

from ultralytics import YOLO


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', type=str, default='yolov8n-seg')
    parser.add_argument('--device', type=str, default=1)
    return parser.parse_args()


def main():
    args = argparser()

    os.makedirs('weights', exist_ok=True)
    model = YOLO(f'weights/{args.checkpoint}.pt', task='segment')
    #model = YOLO(f'weights/{args.checkpoint}.pt', task='여기에 사용할려는 모델 적어주면되 ex) Detect, segment')
    model.train(data='config/yolov8-seg-both.yaml',
                epochs=100,
                batch=32,
                device=int(args.device),
                patience=30,
                project='weights/yolov8/both',
                name=f'{datetime.now().strftime("%m%d")}')


if __name__ == '__main__':
    main()
