import argparse

from ultralytics import YOLO
from ultralytics import settings

from viewer.viewerApp import DemoApp
settings.update({"runs_dir": "runs"})


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', type=str, default='yolov8/ll/1123/weights/best')
    parser.add_argument('-d', '--data', type=str, required=True, help='train val test')
    return parser.parse_args()


def main():
    args = argparser()

    model = YOLO(f'weights/{args.checkpoint}.pt', task='segment')
    DemoApp(args, 'config/yolov8-seg-both.yaml', model)


if __name__ == '__main__':
    main()
