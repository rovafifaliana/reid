from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import cv2

import numpy as np
import tqdm
from ..utils.homography import warp_point, get_perspective_transform, warp_image
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Ellipse

X_SIZE = 105
Y_SIZE = 68

BOX_HEIGHT = (16.5 * 2 + 7.32) / Y_SIZE * 320
BOX_WIDTH = 16.5 / X_SIZE * 320

GOAL = 7.32 / Y_SIZE * 320

GOAL_AREA_HEIGHT = 5.4864 * 2 / Y_SIZE * 320 + GOAL
GOAL_AREA_WIDTH = 5.4864 / X_SIZE * 320

SCALERS = np.array([X_SIZE / 320, Y_SIZE / 320])

def get_field_coordinates(bbox, pred_homo):
    """Computes the warped coordinates of a bounding box
    Arguments:
        bbox: np.array of shape (4,).
        pred_homo: np.array of shape (3,3)
        method: string in {'cv','torch'}, to precise how the homography was predicted
    Returns:
        dst: np.array, the warped coordinates of the bouding box
    Raises:
    """
    #shape_in = 320
    #shape_out = 100
    #ratio = shape_out / shape_in
    #ratio = 1
    x_1 = int(bbox[0])
    y_1 = int(bbox[1])
    x_2 = int(bbox[2])
    y_2 = int(bbox[3])
    x = (x_1 + x_2) / 2.0 #* ratio
    y = max(y_1, y_2) #* ratio
    pts = np.array([float(x), float(y)])
    dst = warp_point(pts, np.linalg.inv(pred_homo))
    return dst


def rgb_template_to_coord_conv_template(rgb_template):
    assert isinstance(rgb_template, np.ndarray)
    assert rgb_template.min() >= 0.0
    assert rgb_template.max() <= 1.0
    rgb_template = np.mean(rgb_template, 2)
    x_coord, y_coord = np.meshgrid(
        np.linspace(0, 1, num=rgb_template.shape[1]),
        np.linspace(0, 1, num=rgb_template.shape[0]),
    )
    coord_conv_template = np.stack((rgb_template, x_coord, y_coord), axis=2)
    return coord_conv_template


def merge_template(img, warped_template):
    valid_index = warped_template[:, :, 0] > 0.0
    overlay = (
        img[valid_index].astype("float32")
        + warped_template[valid_index].astype("float32")
    ) / 2
    new_image = np.copy(img)
    new_image[valid_index] = overlay
    return new_image

def draw_patches(axes, width, height):
    """Draws basic field shapes on an axes
    Arguments:
      axes: matplotlib axes objects.
    Returns:
      axes: matplotlib axes objects.
    Raises:
    """
    # pitch
    axes.add_patch(plt.Rectangle((0, 0), width, height, edgecolor="white", facecolor="none"))

    # half-way line
    axes.add_line(plt.Line2D([width/2, width/2], [height, 0], c="w"))

    # penalty areas
    axes.add_patch(
        plt.Rectangle(
            (width - BOX_WIDTH, (height - BOX_HEIGHT) / 2),
            BOX_WIDTH,
            BOX_HEIGHT,
            ec="w",
            fc="none",
        )
    )
    axes.add_patch(
        plt.Rectangle(
            (0, (height - BOX_HEIGHT) / 2), BOX_WIDTH, BOX_HEIGHT, ec="w", fc="none"
        )
    )

    # # goal areas
    axes.add_patch(
        plt.Rectangle(
            (width - GOAL_AREA_WIDTH, (height - GOAL_AREA_HEIGHT) / 2),
            GOAL_AREA_WIDTH,
            GOAL_AREA_HEIGHT,
            ec="w",
            fc="none",
        )
    )
    axes.add_patch(
        plt.Rectangle(
            (0, (height - GOAL_AREA_HEIGHT) / 2),
            GOAL_AREA_WIDTH,
            GOAL_AREA_HEIGHT,
            ec="w",
            fc="none",
        )
    )

    # goals
    axes.add_patch(plt.Rectangle((width, (height - GOAL) / 2), 1, GOAL, ec="w", fc="none"))
    axes.add_patch(plt.Rectangle((0, (height - GOAL) / 2), -1, GOAL, ec="w", fc="none"))

    # halfway circle
    axes.add_patch(
        Ellipse(
            (width/2, height/2),
            2 * 9.15 / X_SIZE * width,
            2 * 9.15 / Y_SIZE * height,
            ec="w",
            fc="none",
        )
    )

    return axes

def draw_pitch(dpi=100,width=100, height=100, pitch_color="#a8bc95"):
    """Sets up field.
    Arguments:
      dpi: Dots per inch in the field
      pitch_color: Color of the field
    Returns:
      fig,axes: matplotlib fig and axes objects.
    Raises:
    """
    fig = plt.figure(dpi=dpi)
    fig.patch.set_facecolor(pitch_color)

    axes = fig.add_subplot(1, 1, 1)
    axes.set_axis_off()
    axes.set_facecolor(pitch_color)
    axes.xaxis.set_visible(False)
    axes.yaxis.set_visible(False)

    axes.set_xlim(0, width)
    axes.set_ylim(0, height)

    plt.xlim([0,  width])
    plt.ylim([0, height])

    fig.tight_layout(pad=0)
    fig.gca().invert_yaxis()
    draw_patches(axes,width, height)

    return fig, axes

def draw_frame(
    df,
    t,
    dpi=100,
    width=100,
    height=100,
    show_players=True,
    highlight_color=None,
    highlight_player=None,
    text_color="white",
):
    """
    Draws players from time t (in seconds) from a DataFrame df
    """
    fig, ax = draw_pitch(dpi=dpi, width=width, height=height)

    dfFrame = df[df.frame == t]

    if show_players:
        for pid in dfFrame.index:
            # detection of the ball | id = -1
            if pid == -1:
                try:
                    z = dfFrame.loc[pid]["z"]
                except:
                    z = 0
                size = 1.2 + z
                lw = 0.9
                color = "black"
                edge = "white"
                zorder = 100
            else:
                # players detection
                size = 3
                lw = 2

                if pid == highlight_player:
                    color = highlight_color
                else:
                    color = "blue"
                zorder = 20

            ax.add_artist(
                Ellipse(
                    (dfFrame.loc[pid]["x"], dfFrame.loc[pid]["y"]),
                    size / X_SIZE * width,
                    size / Y_SIZE * height,
                    edgecolor="red",
                    linewidth=lw,
                    facecolor=color,
                    alpha=0.8,
                    zorder=zorder,
                )
            )

            try:
                s = str(int(dfFrame.loc[pid]["num"]))
            except ValueError:
                #s = ""
                s = str(int(dfFrame.loc[pid]["id"]))
            text = plt.text(
                dfFrame.loc[pid]["x"],
                dfFrame.loc[pid]["y"],
                s,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=8,
                color=text_color,
                zorder=22,
                alpha=0.8,
            )

            text.set_path_effects(
                [
                    path_effects.Stroke(linewidth=1, foreground=text_color, alpha=0.8),
                    path_effects.Normal(),
                ]
            )
    plt.close()
    return fig, ax, dfFrame

def tlwh_to_tlbr(tlwh):
    ret = np.asarray(tlwh).copy()
    ret[2:] += ret[:2]
    return ret

def saveVideo(args):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap = cv2.VideoCapture(args.path)
    fps = cap.get(cv2.CAP_PROP_FPS)/args.frameEach
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
    template = cv2.imread('Homography/world_cup_template.png')
    template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    template = cv2.resize(template, (width, height))/255.
    template = rgb_template_to_coord_conv_template(template)
    
    for frame in tqdm(range(len(args.keypointsFrames))):
        cap.set(cv2.CAP_PROP_POS_FRAMES,
                args.keypointsFrames[frame][0])
        (_, image) = cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
            pred_homo = get_perspective_transform(
                args.keypointsFrames[frame][2], args.keypointsFrames[frame][1])
            pred_warp = warp_image(cv2.resize(
                template, (args.size, args.size)), pred_homo, out_shape=(args.size, args.size))
            image = merge_template(cv2.resize(
                image/255., (width, height)), cv2.resize(pred_warp, (width, height)))
            # cv2.putText(image,
            #             str(args.keypointsFrames[frame][0]),
            #             (50, 50),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1,
            #             (255, 0, 255),
            #             2,
            #             cv2.LINE_4)
            
            video.write(cv2.cvtColor(np.uint8(image * 255), cv2.COLOR_BGR2RGB))
        except Exception as exception:
            print(
                f"Skipping frame {args.keypointsFrames[frame][0]}")

    # When everything done, release the video capture object
    cap.release()
    video.release()