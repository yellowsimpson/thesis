import cv2
from glob import glob
import numpy as np

img_paths = glob('/Users/mac/Library/Mobile Documents/com~apple~CloudDocs/증명사진/*')
print(img_paths)
img = cv2.imread('/Users/mac/Library/Mobile Documents/com~apple~CloudDocs/증명사진/증명사진.jpg')

