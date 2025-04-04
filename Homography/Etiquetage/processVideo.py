"""
Ce fichier génère des images à partir d'une vidéo.

Usage:
python processVideo.py [chemin des vidéos] [chemin de sauvegarde des images] [nombre d'image par vidéo]
"""


import os
import sys
import cv2
from tqdm import tqdm

folder_path = sys.argv[1]
out_path = sys.argv[2]
nbr_img = sys.argv[3]

def storeImgs(folder_path, vid, out_path, nbr_img):
    """
    Cette fonction prend le chemin des vidéos, extrait les images à intervalles réguliers et les
    enregistre sous forme d'images dans un dossier.
    
    Args:
      folder_path: le chemin où se trouve les vidéos.
      vid: le nom du fichier vidéo.
      out_path: le chemin du répertoire où les images extraites seront enregistrées.
      nbr_img: le nombre d'images à extraire par vidéo.
    """
    cap = cv2.VideoCapture(folder_path+vid)
    x = 0
    step = cap.get(cv2.CAP_PROP_FRAME_COUNT) // nbr_img

    for i in range(0,int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),int(step)):
        cap.set(1,i)
        ret, frame = cap.read()
        cv2.imwrite(out_path+vid.split(".")[0]+"_"+str(x)+".jpg",frame)
        x = x + 1

for vid in tqdm(os.listdir(folder_path)):
    storeImgs(folder_path, vid, out_path, nbr_img)
