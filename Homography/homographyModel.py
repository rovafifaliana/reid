import os

from Homography.src.utils.homography import get_perspective_transform, warp_image
os.environ["SM_FRAMEWORK"] = "tf.keras"
from datetime import datetime
import copy
from Homography.src.utils.masks import _points_from_mask
from Homography.src.optical_flow import extractOpticalFlow
from Homography.src.utils.visualization import  merge_template, rgb_template_to_coord_conv_template
from tqdm import tqdm
import numpy as np
import cv2
from loguru import logger
import tensorflow as tf
from Homography.src.models.keras_models import KeypointDetectorModel

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1
session = tf.compat.v1.Session(config=config)
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


def OpticalFlow(args, cap, reverse):
    TMPFrames = copy.deepcopy(args.keypointsFrames)
    descr = ""
    if reverse:
        interval = reversed(range(1, len(TMPFrames)))
        followingIndex = -1
        descr = "Decremental processing"
    else:
        interval = range(len(TMPFrames)-1)
        followingIndex = 1
        descr = "Incremental processing"

    for src_index in tqdm(interval, desc=descr):
        if (TMPFrames[src_index][3] == 1 or TMPFrames[src_index][3] == 2) and TMPFrames[src_index+followingIndex][3] == 0:
            try:
                # do the process
                cap.set(cv2.CAP_PROP_POS_FRAMES, TMPFrames[src_index][0])
                (_, srcFrame) = cap.read()
                srcFrame = cv2.resize(srcFrame, (args.size, args.size))
                srcFrame = cv2.cvtColor(srcFrame, cv2.COLOR_BGR2RGB)
                cap.set(cv2.CAP_PROP_POS_FRAMES,
                        TMPFrames[src_index+followingIndex][0])
                (_, dstFrame) = cap.read()
                dstFrame = cv2.resize(dstFrame, (args.size, args.size))
                dstFrame = cv2.cvtColor(dstFrame, cv2.COLOR_BGR2RGB)
                transitionVector = extractOpticalFlow(srcFrame, dstFrame)

                if len(TMPFrames[src_index][1]) != 0:
                    newSrc = TMPFrames[src_index][1] + transitionVector
                else:
                    newSrc = TMPFrames[src_index][1]

                if len(TMPFrames[src_index+followingIndex][2]) != 0:
                    for index in np.nonzero((TMPFrames[src_index][2][:, None] == TMPFrames[src_index+followingIndex][2]).all(2).any(1) == 0)[0]:
                        TMPFrames[src_index+followingIndex][1] = np.append(
                            TMPFrames[src_index+followingIndex][1], [newSrc[index]], axis=0).astype(int)
                        TMPFrames[src_index+followingIndex][2] = np.append(
                            TMPFrames[src_index+followingIndex][2], [TMPFrames[src_index][2][index]], axis=0)
                else:
                    TMPFrames[src_index+followingIndex][1] = newSrc
                    TMPFrames[src_index +
                              followingIndex][2] = TMPFrames[src_index][2]
                TMPFrames[src_index+followingIndex][3] = 2

            except Exception as exception:
                logger.error(f"OpticalFlow: {exception}")

    return TMPFrames


def combineOpticalkeypoints(forwardKeypointsFrames, backwardKeypointsFrames, smooth=False):
    combinedOpticalkeypoints = []
    try:
        for index in tqdm(range(len(forwardKeypointsFrames))):
            if forwardKeypointsFrames[index][3] == 1 and backwardKeypointsFrames[index][3] == 1:
                combinedOpticalkeypoints.append(forwardKeypointsFrames[index])
            else:
                if forwardKeypointsFrames[index][3] == 0:
                    combinedOpticalkeypoints.append(
                        backwardKeypointsFrames[index])
                elif backwardKeypointsFrames[index][3] == 0:
                    combinedOpticalkeypoints.append(
                        forwardKeypointsFrames[index])
                else:
                    # Combine replaced frames
                    if len(forwardKeypointsFrames[index][1]) >= len(backwardKeypointsFrames[index][1]):
                        for i in np.nonzero((forwardKeypointsFrames[index][2][:, None] == backwardKeypointsFrames[index][2]).all(2).any(1) == 0)[0]:
                            backwardKeypointsFrames[index][1] = np.append(backwardKeypointsFrames[index][1], [
                                                                            forwardKeypointsFrames[index][1][i]], axis=0)
                            backwardKeypointsFrames[index][2] = np.append(backwardKeypointsFrames[index][2], [
                                                                            forwardKeypointsFrames[index][2][i]], axis=0)

                        combinedOpticalkeypoints.append([backwardKeypointsFrames[index][0], backwardKeypointsFrames[index]
                                                            [1], backwardKeypointsFrames[index][2], 1])
                    else:
                        for i in np.nonzero((backwardKeypointsFrames[index][2][:, None] == forwardKeypointsFrames[index][2]).all(2).any(1) == 0)[0]:
                            forwardKeypointsFrames[index][1] = np.append(forwardKeypointsFrames[index][1], [
                                                                            backwardKeypointsFrames[index][1][i]], axis=0)
                            forwardKeypointsFrames[index][2] = np.append(forwardKeypointsFrames[index][2], [
                                                                            backwardKeypointsFrames[index][2][i]], axis=0)

                        combinedOpticalkeypoints.append(
                            [forwardKeypointsFrames[index][0], forwardKeypointsFrames[index][1], forwardKeypointsFrames[index][2], 1])

            index_sort = np.asarray(sorted(tuple((x[0], x[1], i) for i, x in enumerate(
                combinedOpticalkeypoints[index][2]))))[:, 2]
            combinedOpticalkeypoints[index][1] = combinedOpticalkeypoints[index][1][index_sort]
            combinedOpticalkeypoints[index][2] = combinedOpticalkeypoints[index][2][index_sort]

        if smooth:
            try:
                # smoothing on keypoints
                for index in tqdm(range(1, len(combinedOpticalkeypoints)-1)):
                    combinedOpticalkeypoints[index][1] = (
                        (combinedOpticalkeypoints[index-1][1]+combinedOpticalkeypoints[index][1]+combinedOpticalkeypoints[index+1][1])/3).astype(int)
                # smoothing on optical flow vector
                for index in tqdm(range(len(combinedOpticalkeypoints)-1)):
                    vector = np.median(
                        combinedOpticalkeypoints[index][1]-combinedOpticalkeypoints[index+1][1], axis=0)
                    combinedOpticalkeypoints[index+1][1] = (
                        combinedOpticalkeypoints[index][1]-vector).astype(int)
            except Exception as exception:
                logger.warning(f"smoothingkeypoints: {index} | {exception}")
    except Exception as exception:
        logger.error(f"combineOpticalkeypoints: {exception}")

    return combinedOpticalkeypoints


def moreKeypoints(cap, args, combinedKeypoints, reverse):

    TMPFrames = copy.deepcopy(combinedKeypoints)
    descr = ""
    if reverse:
        interval = reversed(range(1, len(TMPFrames)))
        followingIndex = -1
        descr = "Decremental processing"
    else:
        interval = range(len(TMPFrames)-1)
        followingIndex = 1
        descr = "Incremental processing"

    for src_index in tqdm(interval, desc=descr):
        if (len(TMPFrames[src_index][2]) > len(TMPFrames[src_index+followingIndex][2])):
            try:
                # do the process
                cap.set(cv2.CAP_PROP_POS_FRAMES, TMPFrames[src_index][0])
                (_, srcFrame) = cap.read()
                srcFrame = cv2.resize(srcFrame, (args.size, args.size))
                srcFrame = cv2.cvtColor(srcFrame, cv2.COLOR_BGR2RGB)
                cap.set(cv2.CAP_PROP_POS_FRAMES,
                        TMPFrames[src_index+followingIndex][0])
                (_, dstFrame) = cap.read()
                dstFrame = cv2.resize(dstFrame, (args.size, args.size))
                dstFrame = cv2.cvtColor(dstFrame, cv2.COLOR_BGR2RGB)
                transitionVector = extractOpticalFlow(srcFrame, dstFrame)

                newSrc = TMPFrames[src_index][1] + transitionVector

                for index in np.nonzero((TMPFrames[src_index][2][:, None] == TMPFrames[src_index+followingIndex][2]).all(2).any(1) == 0)[0]:
                    TMPFrames[src_index+followingIndex][1] = np.append(
                        TMPFrames[src_index+followingIndex][1], [newSrc[index]], axis=0).astype(int)
                    TMPFrames[src_index+followingIndex][2] = np.append(
                        TMPFrames[src_index+followingIndex][2], [TMPFrames[src_index][2][index]], axis=0)

            except Exception as exception:
                logger.error("Skipping ", TMPFrames[src_index][0])
                logger.error(f"More keypoints: {exception}")

    return TMPFrames

def homography(args, save_video):
    full_model = KeypointDetectorModel(
        input_shape=(args.size, args.size), activation=args.activation)

    if args.weights is not None:
        full_model.load_weights(args.weights)

    cap = cv2.VideoCapture(args.path)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        logger.error("Error opening video stream or file")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
    template = cv2.imread('Homography/world_cup_template.png')
    template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    template = cv2.resize(template, (width, height))/255.
    template = rgb_template_to_coord_conv_template(template)
    logger.info(
        f"STEP 1 : Homography model with {int(cap.get(cv2.CAP_PROP_FRAME_COUNT)/args.frameEach)}")
    start_time = datetime.now()
    x = 0
    detectedHomography = 0
    # Read until video is completed
    while (cap.isOpened()):

        # Capture frame-by-frame
        (_, frame) = cap.read()
        if frame is None:
            break
        if (x % args.frameEach == 0):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pr_mask = full_model(frame)
            src, dst = _points_from_mask(pr_mask[0])
            if (len(src) >= args.keypointTreshold):
                try:
                    args.keypointsFrames.append([x, src, dst, 1])
                    detectedHomography+=1
                except:
                    logger.warning(f"STEP 1 : Skipping frame {x}")
            else:
                # logger.error("il manque des keypoints !")
                args.keypointsFrames.append([x, src, dst, 0])

            # plt.imsave(f"output/{x}_image.png", frame)

        x += 1
    logger.info(f"Homography Ratio:{detectedHomography/len(args.keypointsFrames)}")
    end_time = datetime.now()
    step1_duration = end_time - start_time

    if args.useOpticalFlow:

        # Now use optical flow to replace all missing homographies
        logger.info("STEP 2 : Optical Flow")
        start_time = datetime.now()
        logger.info(f"2.1 Processing frames")
        forwardOpticalkeypointsFrames = OpticalFlow(args, cap, reverse=False)

        # do the same but in the other way
        backwardOpticalkeypointsFrames = OpticalFlow(args, cap, reverse=True)

        # combine the results
        logger.info("2.2 Combining keypoints")
        combinedOpticalkeypoints = combineOpticalkeypoints(
            forwardOpticalkeypointsFrames, backwardOpticalkeypointsFrames)

        # smooth the results by adding more keypoints
        logger.info("STEP 3 : Increasing keypoints")

        combinedOpticalkeypoints = combineOpticalkeypoints(
            combinedOpticalkeypoints, combinedOpticalkeypoints, smooth=True)
        
        combinedOpticalkeypoints = moreKeypoints(cap,args,
            combinedOpticalkeypoints, reverse=False)
        combinedOpticalkeypoints = moreKeypoints(cap,args,
            combinedOpticalkeypoints, reverse=True)
        args.keypointsFrames = combinedOpticalkeypoints
        for values in combinedOpticalkeypoints:
            pred_homo = get_perspective_transform(values[2],values[1])
            args.homographies[values[0]] = pred_homo
        end_time = datetime.now()
        step2_duration = end_time - start_time
    else:
        combinedOpticalkeypoints = args.keypointsFrames

    if (save_video):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)/args.frameEach

        # video = cv2.VideoWriter('processed_output_guinee.mp4', fourcc, fps, (width, height))
        video = cv2.VideoWriter('Untitled_output.mp4', fourcc, fps, (width, height))

        for frame in tqdm(range(len(combinedOpticalkeypoints))):
            cap.set(cv2.CAP_PROP_POS_FRAMES,
                    combinedOpticalkeypoints[frame][0])
            (_, image) = cap.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            try:
                pred_homo = get_perspective_transform(
                    combinedOpticalkeypoints[frame][2], combinedOpticalkeypoints[frame][1])
                pred_warp = warp_image(cv2.resize(
                    template, (args.size, args.size)), pred_homo, out_shape=(args.size, args.size))
                # image = merge_template(cv2.resize(
                #     image/255., (width, height)), cv2.resize(pred_warp, (width, height)))
                # cv2.putText(image,
                #             str(combinedOpticalkeypoints[frame][0]),
                #             (50, 50),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1,
                #             (255, 0, 255),
                #             2,
                #             cv2.LINE_4)
                mask = cv2.cvtColor((pred_warp > 0.1).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                mask = cv2.resize(mask, (width, height))
                masked_frame = cv2.bitwise_and(image, image, mask=mask)
                # video.write(cv2.cvtColor(np.uint8(image * 255), cv2.COLOR_BGR2RGB))
                video.write(cv2.cvtColor(np.uint8(masked_frame), cv2.COLOR_BGR2RGB))

            except Exception as exception:
                print(
                    f"Skipping frame {combinedOpticalkeypoints[frame][0]}")

    # When everything done, release the video capture object
    cap.release()
    video.release()
    logger.info('Duration (AI model) step 1: {}'.format(step1_duration))
    if args.useOpticalFlow:
        logger.info('Duration (Optical flow) step 2: {}'.format(step2_duration))