import os
import sys
import time
import numpy as np
import skimage.io
import cv2
import pandas as pd

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
#  
# A quick one liner to install the library 
# !pip install git+https://github.com/waleedka/coco.git#subdirectory=PythonAPI

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import coco #a slightly modified version

from mrcnn.evaluate import build_coco_results, evaluate_coco
from mrcnn.dataset import MappingChallengeDataset
from mrcnn import visualize

import zipfile
import urllib.request
import shutil
import glob
import tqdm
import random


ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils


PRETRAINED_MODEL_PATH = os.path.join(ROOT_DIR,"data/" "pretrained_weights.h5")
# PRETRAINED_MODEL_PATH = os.path.join(ROOT_DIR,"data/\" \"pretrained_weights.h5\"),
LOGS_DIRECTORY = os.path.join(ROOT_DIR, "logs")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
IMAGE_DIR = os.path.join(ROOT_DIR, "data", "test", "images")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output",)
OUTPUT_EACH_BUILDING = os.path.join(ROOT_DIR, "hasil",)
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)]

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 5
    NUM_CLASSES = 1 + 1  # 1 Background + 1 Building
    IMAGE_MAX_DIM=320
    IMAGE_MIN_DIM=320
    NAME = "crowdai-mapping-challenge"
config = InferenceConfig()
config.display()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model_path = PRETRAINED_MODEL_PATH

# or if you want to use the latest trained model, you can use : 
# model_path = model.find_last()[1]
model.load_weights(model_path, by_name=True)

class_names = ['BG', 'building'] # In our case, we have 1 class for the background, and 1 class for building

# Images reload
file_names = next(os.walk(IMAGE_DIR))[2]
file_name = "000000000012.jpg"

# Load the image
# image = cv2.imread(os.path.join(IMAGE_DIR, file_name))
image = skimage.io.imread(os.path.join(IMAGE_DIR, file_name))

# Detect objects in the image
predictions = model.detect([image]*config.BATCH_SIZE, verbose=1)
p = predictions[0]

# Get the class IDs of the detected objects
class_ids = p['class_ids']

# Get the indices of the detected buildings
building_indices = [i for i, class_id in enumerate(class_ids) if class_id == class_names.index("building")]

results = []
total_rooftop_area = 0

# Check if there are any buildings detected
if len(building_indices) == 0:
    print("No buildings detected.")
else:
    # Draw bounding box and mask overlay on original image
    for i in building_indices:
        # Get the bounding box coordinates of the building
        y1, x1, y2, x2 = p['rois'][i]

        # Draw the bounding box on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Get the mask for the building
        mask = p['masks'][:,:,i]

        # Draw the mask overlay on the image
        mask_overlay = (0.3 * image + 0.7 * (mask[..., None] > 0) * [255, 255, 255]).astype('uint8')
        mask_overlay[mask == 0] = [0, 0, 0]  # Set the unmasked part to black
        image = cv2.addWeighted(image, 0.5, mask_overlay, 0.5, 0)

        # Crop the building from the image
        building = image[y1:y2, x1:x2]

        # Save the building as a separate file
        output_path = os.path.join(OUTPUT_EACH_BUILDING, f"./Building-{file_name}-{i}.jpg")
        cv2.imwrite(output_path, building)

        image2 = cv2.cvtColor(building, cv2.COLOR_BGR2GRAY)

        # Apply bitwise thresholding to convert the image to black and white
        _, bw_image = cv2.threshold(image2, 100, 255, cv2.THRESH_BINARY)

        n_black_pix = np.sum(bw_image == 0)
        n_white_pix = np.sum(bw_image == 255)
        # print('Number of black pixels:', n_black_pix)
        # print('Number of white pixels:', n_white_pix)

        # print('Rooftop area:', n_white_pix)
        # print('Rooftop percentage:', n_white_pix/(n_black_pix+n_white_pix))
        # Compute the rooftop area and percentage
        rooftop_area = n_white_pix
        rooftop_percentage = n_white_pix / (n_black_pix + n_white_pix)

        # Add the results to the list
        results.append({
            "File Name": file_name,
            "Building Index": i,
            "Rooftop Area": rooftop_area,
            "Rooftop Percentage": rooftop_percentage
            })

        # Calculate rooftop area
        total_rooftop_area += rooftop_area

# Create a pandas dataframe from the results list
df = pd.DataFrame(results)

# Add total rooftop area to the dataframe
df.loc[len(df)] = ["Total", "", total_rooftop_area, ""]

# Print the dataframe
print(df)

def display_instances(image, boxes, masks, class_ids, class_names, scores):
    """
    Display instances on the image.
    """
    # Create a black background image
    output_image = np.zeros(image.shape, dtype=np.uint8)

    # Loop through each instance and draw it on the output image
    for i in range(len(class_ids)):
        # Get the class ID and mask for this instance
        class_id = class_ids[i]
        mask = masks[:, :, i]

        # Get the color for this class
        color = COLORS[class_id]

        # Draw the mask on the output image
        alpha = 0.3
        for c in range(3):
            output_image[:, :, c] = np.where(mask, output_image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                              output_image[:, :, c])

    return output_image

# Get the file names in the input directory
file_names = next(os.walk(IMAGE_DIR))[2]

# Load a random image from the input directory
random_image = skimage.io.imread(os.path.join(IMAGE_DIR, "000000000012.jpg"))

# Make predictions on the image
predictions = model.detect([random_image]*config.BATCH_SIZE, verbose=1)
p = predictions[0]

# # Save the output image to a file
# file_name = "output.jpg"
# output_path = os.path.join(OUTPUT_DIR, file_name)
# if output_image.any():
#     output_path = os.path.join(OUTPUT_DIR, file_name)
#     skimage.io.imsave(output_path, output_image)
# else:
#     print("No instances detected in image:", file_name)

def area_in_m2(area_in_pixel,scale):
    return area_in_pixel*scale

def number_pv_module(area, module_width, module_length):
    return int(area/(module_length*module_width))

def global_iradiation(location_spesific_solar_irradiance, total_rooftop_area):
    return location_spesific_solar_irradiance*total_rooftop_area


def maximum_pv_system_capacity(area, module_width, module_length, capacity_per_module):
    pv_number = number_pv_module(area, module_width, module_length)
    print('pv number:', pv_number)
    return pv_number*capacity_per_module

def yearly_energy(area, location_spesific_solar_irradiance,efficiency_per_module,system_losses_percentage):
    iradiation = global_iradiation(location_spesific_solar_irradiance, area)
    print('global iradiation:', iradiation)
    return iradiation*efficiency_per_module*system_losses_percentage

def calculate_potential(area_in_pixel,scale,module_width, module_length, capacity_per_module,location_spesific_solar_irradiance,efficiency_per_module,system_losses_percentage):
    area = area_in_m2(area_in_pixel,scale)
    capacity = maximum_pv_system_capacity(area, module_width, module_length, capacity_per_module)
    print('Maximum PV system capacity:', capacity)
    energy = yearly_energy(area, location_spesific_solar_irradiance,efficiency_per_module,system_losses_percentage)
    print('Maximum PV system capacity:', energy)
    print('Total Area in Pixel: ',total_rooftop_area )



#TEST
area_in_pixel = total_rooftop_area 
scale = 0.1
module_width = 0.5
module_length = 0.5
capacity_per_module = 500
location_spesific_solar_irradiance = 30
efficiency_per_module = 0.75
system_losses_percentage = 0.8

calculate_potential(area_in_pixel,scale,module_width, module_length, capacity_per_module,location_spesific_solar_irradiance,efficiency_per_module,system_losses_percentage)

# Display the instances on the image
visualize.display_instances(random_image, p['rois'], p['masks'], p['class_ids'], 
                            class_names, p['scores'])