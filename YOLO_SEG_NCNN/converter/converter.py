import argparse

import cv2
from glob import glob

import numpy as np
from tqdm import tqdm


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--label', type=str, required=True, help='da ll')
    return parser.parse_args()


args = argparser()


def convert_to_yolo_format(contour, image_width, image_height):
    normalized_contour = []
    for point in contour:
        x, y = point[0]
        normalized_x = x / image_width
        normalized_y = y / image_height
        normalized_contour.extend([normalized_x, normalized_y])
    return normalized_contour


def find_contours_coordinates(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    bs = 1
    h, w = image.shape
    bordered_image = np.zeros(image.shape, dtype=image.dtype)
    bordered_image[bs:h - bs, bs:w - bs] = image[bs:h - bs, bs:w - bs]

    # da
    if args.label == 'da':
        _, thresh = cv2.threshold(bordered_image, 126, 255, cv2.THRESH_BINARY)
    # ll
    elif args.label == 'll':
        _, thresh = cv2.threshold(bordered_image, 254, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def convert(label_path, output_path):
    for label in tqdm(label_path):
        contours = find_contours_coordinates(label)

        with open(output_path + '{}'.format(label.split('/')[-1].replace('.png', '.txt')), 'w') as file:
            for contour in contours:
                yolo_format = convert_to_yolo_format(contour, 1280, 720)
                line = '{} '.format(0) + ' '.join(map(lambda x: f"{x:.3f}", yolo_format)) + '\n'
                file.write(line)


if __name__ == '__main__':
    path = '/Users/mac/Desktop/BDD100k'

    # label_path = glob(path + f'/{args.label}/train/*')
    # print(path + f'/{args.label}/train/*')
    # exit()
    # convert(label_path, path + '/train/')

    label_path = glob(path + f'/{args.label}/val/*')
    # print(path + f'/{args.label}/train/*')
    # exit()
    convert(label_path, path + '/val/')
