seed=123
from keras import backend as K

import numpy as np
np.random.seed(seed)
import tensorflow as tf

tf.random.set_random_seed(seed)

import random
random.seed(seed)

import skimage.io 
from skimage import img_as_ubyte
import cv2
import pandas as pd
import model as modellib
import os
from os import path, mkdir
from tqdm import tqdm
import my_functions as f


#######################################################################################
## SET UP CONFIGURATION
from config import Config

class BowlConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "Inference"

    IMAGE_RESIZE_MODE = "pad64" ## tried to modfied but I am using other git clone
    ## No augmentation
    ZOOM = False
    ASPECT_RATIO = 1
    MIN_ENLARGE = 1
    IMAGE_MIN_SCALE = False ## Not using this

    IMAGE_MIN_DIM = 512 # We scale small images up so that smallest side is 512
    IMAGE_MAX_DIM = False

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    DETECTION_MAX_INSTANCES = 512
    DETECTION_NMS_THRESHOLD =  0.2
    DETECTION_MIN_CONFIDENCE = 0.9

    LEARNING_RATE = 0.001
    
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 # background + nuclei

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 , 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 600

    USE_MINI_MASK = True

def preprocess_inputs(x):
    x = np.asarray(x, dtype='float32')
    x /= 127.5
    x -= 1.
    return x

def bgr_to_lab(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(17, 17))
    lab = clahe.apply(lab[:, :, 0])
    if lab.mean() > 127:
        lab = 255 - lab
    return lab[..., np.newaxis]

def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None,
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()
    return masked_image


inference_config = BowlConfig()
inference_config.display()
#######################################################################################


ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

model_path = "../maskrcnn.h5"

data_folder = path.join('../..', 'data')

masks_folder = path.join(data_folder, 'masks_all')
images_folder = path.join(data_folder, 'images_all')
labels_folder = path.join(data_folder, 'labels_all')

train_pred = path.join('../..', 'predictions', 'maskrcnn_oof')

df = pd.read_csv(path.join(data_folder, 'folds.csv'))

all_ids = []
all_images = []
all_masks = []

print("Loading weights from ", model_path)


import time
start_time = time.time()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)
model.load_weights(model_path, by_name=True)


ImageId_d = []
EncodedPixels_d = []

if not path.isdir(train_pred):
    mkdir(train_pred)

all_ids = df['img_id'].values
all_sources = df['source'].values

for i in tqdm(range(len(all_ids))):
    img_id = all_ids[i]
    img = cv2.imread(path.join(images_folder, '{0}.png'.format(img_id)), cv2.IMREAD_COLOR)
    all_images.append(img)

for it in range(4):
    val_idx = df[(df['fold'] == it)].index.values
    print('Predicting fold', it)
    for i in tqdm(val_idx):
        final_mask = None
        for scale in range(3):
            fid = all_ids[i]
            img = all_images[i]
            if final_mask is None:
                final_mask = np.zeros((img.shape[0], img.shape[1], 3))
            if scale == 1:
                img = cv2.resize(img, None, fx=0.75, fy=0.75)
            elif scale == 2:
                img = cv2.resize(img, None, fx=1.25, fy=1.25)
            elif scale == 3:
                img = cv2.resize(img, None, fx=1.5, fy=1.5)

            x0 = 16
            y0 = 16
            x1 = 16
            y1 = 16
            if (img.shape[1] % 32) != 0:
                x0 = int((32 - img.shape[1] % 32) / 2)
                x1 = (32 - img.shape[1] % 32) - x0
                x0 += 16
                x1 += 16
            if (img.shape[0] % 32) != 0:
                y0 = int((32 - img.shape[0] % 32) / 2)
                y1 = (32 - img.shape[0] % 32) - y0
                y0 += 16
                y1 += 16
            img0 = np.pad(img, ((y0, y1), (x0, x1), (0, 0)), 'symmetric')

            img0 = np.concatenate([img0, bgr_to_lab(img0)], axis=2)

            inp0 = []
            inp1 = []
            for flip in range(2):
                for rot in range(4):
                    if flip > 0:
                        img = img0[::-1, ...]
                    else:
                        img = img0
                    if rot % 2 == 0:
                        inp0.append(np.rot90(img, k=rot))
                    else:
                        inp1.append(np.rot90(img, k=rot))

            inp0 = np.asarray(inp0)
            inp0 = preprocess_inputs(inp0)
            inp1 = np.asarray(inp1)
            inp1 = preprocess_inputs(inp1)

            mask = np.zeros((img0.shape[0], img0.shape[1], 3))

            results0 = model.detect([inp0], verbose=0)
            results1 = model.detect([inp1], verbose=0)

            pred0 = display_instances(img, results0['rois'], results0['masks'], results0['class_ids'],
                    ['BG', 'nucleus'], results0['scores'],
                    show_bbox=False, show_mask=True)
            j = -1
            for flip in range(2):
                for rot in range(4):
                    j += 1
                    if rot % 2 == 0:
                        pr = np.rot90(pred0[int(j / 2)], k=(4 - rot))
                    else:
                        pr = np.rot90(pred1[int(j / 2)], k=(4 - rot))
                    if flip > 0:
                        pr = pr[::-1, ...]
                    mask += pr

            mask /= 8
            mask = mask[y0:mask.shape[0] - y1, x0:mask.shape[1] - x1, ...]
            if scale > 0:
                mask = cv2.resize(mask, (final_mask.shape[1], final_mask.shape[0]))
            final_mask += mask
        final_mask /= 3
        final_mask = final_mask * 255
        final_mask = final_mask.astype('uint8')
        cv2.imwrite(path.join(train_pred, '{0}.png'.format(fid)), final_mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        print('{0}.png'.format(fid))
end_time = time.time()
ellapsed_time = (end_time-start_time)/3600
print('Time required to infer ', ellapsed_time, 'hours')

