import torch
import utils
import tqdm
import cv2
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from loguru import logger
import os.path as osp
from tools.mc_demo import Predictor
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from tracker.mc_bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer



def get_yolox_model(exp, args):
    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(args.OUTPUT_DIR, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(args.OUTPUT_DIR, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    
    return predictor

def initialize_video_writer(video_path, out_video):
    sequence = cv2.VideoCapture(video_path)
    fps = sequence.get(cv2.CAP_PROP_FPS)
    print("\n_n fps = ", fps, "\n\n")
    frame_width = int(sequence.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(sequence.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(sequence.get(cv2.CAP_PROP_FRAME_COUNT))
    out_sequence = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    return sequence, out_sequence, fps, frame_width, frame_height, n_frames

def get_grass_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40]) 
    upper_green = np.array([80, 255, 255]) 
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    grass_color = cv2.mean(img, mask=mask)
    return grass_color[:3]

def equalize_histogram(player_img):
    hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    equalized_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return equalized_img

def normalize_colors(player_img):
    # lab_img = cv2.cvtColor(player_img, cv2.COLOR_BGR2LAB)
    normalized_img = cv2.normalize(player_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return normalized_img

def darken_image(img, factor=0.5):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    darkened_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return darkened_img

def refine_mask(player_img, grass_hsv):
    hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([grass_hsv[0, 0, 0] - 10, 40, 40])
    upper_green = np.array([grass_hsv[0, 0, 0] + 10, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.bitwise_not(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    upper_mask = np.zeros(player_img.shape[:2], np.uint8)
    upper_mask[0:player_img.shape[0]//2, 0:player_img.shape[1]] = 255
    mask = cv2.bitwise_and(mask, upper_mask)
    return mask

def resize_image(img, target_size):
    """
    Resize image to the target size while maintaining the aspect ratio.
    """
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized_img

def get_kits_colors(players, grass_hsv=None, frame=None):
    kits_colors = []
    if grass_hsv is None:
        grass_color = get_grass_color(frame)
        grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)
    for player_img in players:
        if len(player_img.shape) == 2:
            player_img = cv2.cvtColor(player_img, cv2.COLOR_GRAY2BGR)
        if player_img.size == 0:
            continue
        equalized_img = equalize_histogram(player_img)
        darkened_img = darken_image(equalized_img, factor = 10)
        normalized_img = normalize_colors(darkened_img)
        mask = refine_mask(normalized_img, grass_hsv)
        kit_color = np.array(cv2.mean(normalized_img, mask=mask)[:3])
        kits_colors.append(kit_color)
    return kits_colors

def get_kits_classifier(kits_colors):
    kits_kmeans = KMeans(n_clusters=3, random_state=42, max_iter=3000, algorithm='lloyd')
    kits_kmeans.fit(kits_colors)
    return kits_kmeans

def classify_kits(kits_classifier, kits_colors):
    team = kits_classifier.predict(kits_colors)
    return team

def cluster_players(detections, raw_img, frame_idx, grass_hsv_value, kits_classifier):
    player_imgs, player_boxes = get_players_boxes(detections, raw_img)
    if len(player_imgs) == 0:
        return "None"
    kits_colors = get_kits_colors(player_imgs, grass_hsv=grass_hsv_value, frame=raw_img)

    if int(frame_idx) == 1 and kits_colors:
        kits_classifier = get_kits_classifier(kits_colors)
    grass_color = get_grass_color(raw_img)
    grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)

    labels = []
    for kit_color in kits_colors:
        team = classify_kits(kits_classifier, [kit_color])
        if team == 0:
            labels.append(0)
        elif team == 2:
            labels.append(2)
        else:
            labels.append(1)
    
    return labels, kits_classifier, grass_hsv