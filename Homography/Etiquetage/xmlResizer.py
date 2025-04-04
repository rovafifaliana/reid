import xml.etree.ElementTree as ET
import sys
import os
from tqdm import tqdm

coef = 512 / 320 #320 / 512

folder_path = sys.argv[1]
out_path = sys.argv[2]

for d in tqdm(os.listdir(folder_path)):
    # parsing from the string.
    mytree = ET.parse(folder_path+d)
    myroot = mytree.getroot()

    for o in myroot.iter("object"):
        for k in o.find("keypoints"):
            if k.tag == "x1" or k.tag == "y1":
                k.text = str(round(float(k.text) * coef))
        for k in o.find("bndbox"):
                k.text = str(round(float(k.text) * coef))

    myroot.find("size").find("width").text = str(512) #320
    myroot.find("size").find("height").text = str(512) #320

    mytree.write(out_path+d, encoding="utf-8")
