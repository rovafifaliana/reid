import cv2
import sys
import os
from tqdm import tqdm

folder_path = sys.argv[1]
out_path = sys.argv[2]

for d in tqdm(os.listdir(folder_path)):

    img = cv2.imread(folder_path+d)
    img = cv2.resize(img, (512,512))
    cv2.imwrite(out_path+d, img)