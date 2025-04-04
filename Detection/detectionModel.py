# from pathlib import Path
# import sys
# path_root = Path(__file__).parents[0]
# sys.path.append(str(path_root))

import tqdm
import torch
import cv2
import os.path as osp
from loguru import logger
from tools.mc_demo import Predictor
from yolox.exp import get_exp
from yolox.utils.visualize import plot_tracking
from tracker.tracking_utils.timer import Timer
from yolox.utils import fuse_model, get_model_info
from tracker.mc_bot_sort import BoTSORT
from paddleocr import PaddleOCR
from ultralytics import YOLO
import numpy as np
from sklearn.cluster import KMeans
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict

timer = Timer()

def get_yolox_model(exp, args):
    print("\nexp: ", exp, "\n")
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

def get_players_boxes(results, frame):
    players_imgs = []

    for box in results:
        x1, y1, w, h = map(int, box[:4])
        x2, y2 = x1 + w, y1 + h
        x1 = 0 if x1 < 0 else x1
        y1 = 0 if y1 < 0 else y1
        player_img = frame[y1:y2, x1:x2]
        players_imgs.append(player_img)

    return players_imgs

def equalize_histogram(player_img):
    hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    equalized_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return equalized_img

def normalize_colors(player_img):
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

def contour_amplification(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edged, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    new_frame = frame.copy()
    cv2.drawContours(new_frame, contours, -1, (0, 0, 255), 1, cv2.LINE_AA)
    return new_frame

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
    kits_kmeans = KMeans(n_clusters=3, random_state=42, max_iter=30000, algorithm='lloyd')
    kits_kmeans.fit(kits_colors)
    return kits_kmeans

def classify_kits(kits_classifier, kits_colors):
    team = kits_classifier.predict(kits_colors)
    return team

def cluster_players(detections, raw_img, frame_idx, grass_hsv_value, kits_classifier):
    player_imgs = get_players_boxes(detections, raw_img)
    if len(player_imgs) == 0:
        return "None"
    kits_colors = get_kits_colors(player_imgs, grass_hsv=grass_hsv_value, frame=raw_img)

    if int(frame_idx) == 0 and kits_colors:
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

def process_frame(model, ball_model, frame, frame_idx, grass_hsv_value, kits_classifier, exp):
    new_frame = contour_amplification(frame)
    frame_rgb = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
    outputs, img_info = model.inference(frame_rgb, timer)
    scale = min(exp.test_size[0] / float(img_info['height'], ), exp.test_size[1] / float(img_info['width']))
    detections = []
    if outputs[0] is not None:
        outputs = outputs[0].cpu().numpy()
        detections = outputs[:, :7]
        detections[:, :4] /= scale
    else:
        "None"
    output = get_sliced_prediction(
        frame,
        ball_model,
        slice_height=frame.shape[0]//6,
        slice_width=frame.shape[1]//6,
    )
    if len(output.object_prediction_list) == 0:
        full_detections = detections
    else:
        output.object_prediction_list = [output.object_prediction_list[0]] if len(output.object_prediction_list)>0 else output.object_prediction_list
        boxes = np.array(output.object_prediction_list[0].bbox.to_xyxy()).reshape(1, -1)
        conf_score = output.object_prediction_list[0].score.value
        label = output.object_prediction_list[0].category.id
        reshaped_score = np.array(conf_score).reshape(-1, 1)
        reshaped_label = np.repeat(label+3, boxes.shape[0]).reshape(-1, 1)
        ball_detection = np.column_stack((boxes, reshaped_score, reshaped_score, reshaped_label))

        if np.array(detections).size == 0:
            full_detections = ball_detection
        else:
            full_detections = np.vstack((detections, ball_detection))
    
    return full_detections, frame

def get_kits_number(detections, raw_img, ocr):
    ocr_results = []
    player_imgs = get_players_boxes(detections, raw_img)
    for player_img in player_imgs:
        ocr_result = ()
        resized_img = resize_image(player_img, 640)
        roi = resized_img
        if not isinstance(roi, np.ndarray):
            continue
        result = ocr.ocr(roi)
        default_text = ''
        if result[0]:
            for line in result[0]:
                text = line[1][0]
                score_text = line[1][1]
                if text.isdigit() and len(text) <= 2 and score_text >= 0.8: 
                    ocr_result = (text, score_text)
                else:
                    ocr_result = (default_text, 0)
        else:
            ocr_result = (default_text, 0)
        ocr_results.append(ocr_result)
    return ocr_results

def update_tracker(args, tracker, detections, final_outputs, frame_idx, raw_img):

    online_targets = tracker.update(detections, raw_img)
    online_tlwhs = []
    online_ids = []
    online_scores = []
    online_labels = []
    
    for t in online_targets:
        tlwh = t.tlwh
        tid = t.track_id
        if tlwh[2] * tlwh[3] > args.min_box_area:
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_scores.append(t.score)
            online_labels.append(t.cls)
            final_outputs.append(
                [frame_idx,tid,round(tlwh[0],3),round(tlwh[1],3),round(tlwh[2],3),round(tlwh[3],3),round(t.score,3), t.cls]
            )
    return online_tlwhs, online_ids

def complete_output(final_outputs, team_label, kit_number):
    for output, label, kit in zip(final_outputs, team_label, kit_number):
        output.extend([label, kit[0]])

def classify(args):
    exp = get_exp(args.exp_file, args.name)
    args.ablation = False
    args.mot20 = not args.fuse_score
    model = get_yolox_model(exp, args)
    tracker = BoTSORT(args, frame_rate = args.fps)

    # ball_model = YOLO("runs/detect/train/weights/best.pt")
    # ball_model.to(args.device)

    # ball_model = AutoDetectionModel.from_pretrained(
    #     model_type="yolov8",
    #     model_path="runs/detect/train/weights/best.pt",
    #     confidence_threshold=0.5,
    #     device=args.device
    # )
    ball_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path="/Users/rovafifaliana/Documents/Sfeira/SSD-Football-Tracking/BoTSORT/pretrained/yolov8s.pt",
        confidence_threshold=0.5,
        device=args.device
    )
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, show_log=False)
    
    grass_hsv_value = None
    kits_classifier = None
    
    sequence = cv2.VideoCapture(args.path)
    out_sequence = cv2.VideoWriter("sfeira_output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (int(sequence.get(cv2.CAP_PROP_FRAME_WIDTH)), int(sequence.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    n_frames = int(sequence.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Processing video: {args.path}')
    pbar = tqdm.tqdm(total=n_frames)
    frame_idx = 0
    
    try:
        while sequence.isOpened():
            ret, frame = sequence.read()
            if not ret:
                break
            final_outputs = []
            if (frame_idx % args.frameEach == 0):
                #frame_idx = sequence.get(cv2.CAP_PROP_POS_FRAMES)
                if not isinstance(process_frame(model, ball_model, frame, frame_idx, grass_hsv_value, kits_classifier, exp), str):
                    new_results, raw_img = process_frame(model, ball_model, frame, frame_idx, grass_hsv_value, kits_classifier, exp)
                    if len(new_results) == 0:
                        pbar.update(1)
                    else:
                        args.online_tlwhs, args.online_ids = update_tracker(args, tracker, new_results, final_outputs, frame_idx, raw_img)
                        labels, kits_classifier, grass_hsv = cluster_players(args.online_tlwhs, raw_img, frame_idx, grass_hsv_value, kits_classifier)
                        ocr_results = get_kits_number(args.online_tlwhs, raw_img, ocr)
                        grass_hsv_value = grass_hsv
                        complete_output(final_outputs, labels, ocr_results)
                        args.final_outputs.extend(final_outputs)
                        timer.toc()
                        frame = plot_tracking(raw_img, args.online_tlwhs, args.online_ids, frame_id=frame_idx + 1, fps=1. /timer.average_time)
                        out_sequence.write(frame)
                        pbar.update(1)
                else:
                    pbar.update(1)
            frame_idx += 1

    finally:
        pbar.close()
        out_sequence.release()