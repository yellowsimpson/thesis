import argparse
import os
from datetime import datetime

from ultralytics import YOLO


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', type=str, required=True)
    return parser.parse_args()


def export_yolo(args):
    model = YOLO(f'weights/{args.checkpoint}.pt')
    model.export(format="onnx", opset=13)


def main():
    args = argparser()
    export_yolo(args)


if __name__ == '__main__':
    main()
