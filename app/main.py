import os
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import base64
import shutil

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

from notebooks.samples.coco import coco #a slightly modified version

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

PRETRAINED_MODEL_PATH = os.path.join(ROOT_DIR,"notebooks/data/" "pretrained_weights.h5")
# PRETRAINED_MODEL_PATH = os.path.join(ROOT_DIR,"data/\" \"pretrained_weights.h5\"),
LOGS_DIRECTORY = os.path.join(ROOT_DIR, "logs")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
IMAGE_DIR = os.path.join(ROOT_DIR, "uploads")
OUTPUT_DIR = os.path.join(ROOT_DIR, "cropimage",)
OUTPUT_EACH_BUILDING = os.path.join(ROOT_DIR, "cropimage", "hasil",)
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)]

app = FastAPI()

# Configure CORS middleware
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

model.load_weights(model_path, by_name=True)

class_names = ['BG', 'building'] # In our case, we have 1 class for the background, and 1 class for building

xfilename = ""
scale = 0
width = 0
length = 0
capacitypm = 0
irradiance = 0
efficiency = 0
system_losses = 0
total_rooftop_area = 0
energy = 0
capacity = 0
max_pv = 0
global_irradiation = 0
total_area = 0

@app.get('/')
async def root():
    return {"message": "GIZ Project"}

# POST Image
@app.post('/uploadimage')
async def create_upload_file(scalex:float= Form(...),widthx:float= Form(...),lengthx:float= Form(...),capacityx:float= Form(...),irradiancex:float= Form(...),efficiencyx:float= Form(...),system_lossesx:float= Form(...),file: UploadFile = File(...)):
    global energy, capacity,max_pv,global_irradiation,total_area,total_rooftop_area,xfilename,scale,width,length,capacitypm,irradiance,efficiency,system_losses
    xfilename = file.filename
    file_name = xfilename
    scale = scalex
    width = widthx
    length = lengthx
    capacitypm = capacityx
    irradiance = irradiancex
    efficiency = efficiencyx
    system_losses = system_lossesx
    total_rooftop_area = 0
    energy = 0
    capacity = 0
    max_pv = 0
    global_irradiation = 0
    total_area = 0
    
    try:
        # Create directory for the upload files
        # os.makedirs("generate", exist_ok=True)

        # Save the uploaded file to the server
        with open(f"uploads/{xfilename}", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Open the image using Pillow
        with Image.open(f"uploads/{xfilename}") as image:
            image.thumbnail((128, 128))
            thumbnail_filename = f"uploads/{xfilename.split('.')[0]}_thumbnail.jpg"
            image.save(thumbnail_filename, format="JPEG")

        # Read the thumbnail as Based64-encoded string
        with open(thumbnail_filename, "rb") as f:
            thumbnail_bytes = f.read()
            thumbnail_str = base64.b64encode(thumbnail_bytes).decode("utf-8")

        # Process image to Detect -> run 
        random_image = skimage.io.imread(os.path.join(IMAGE_DIR, f"{xfilename.split('.')[0]}.jpg"))

        predictions = model.detect([random_image]*config.BATCH_SIZE, verbose=1) # We are replicating the same image to fill up the batch_size

        p = predictions[0]
        # Save image with Masking
        visualize.save_image(os.path.join(IMAGE_DIR, f"{xfilename.split('.')[0]}.jpg"), f"genimage/{xfilename.split('.')[0]}.jpg", p['rois'], p['masks'],p['class_ids'], class_names, p['scores'])

        # Load the image
        image = cv2.imread(os.path.join(IMAGE_DIR, file_name))

        # Detect objects in the image
        predictions = model.detect([image]*config.BATCH_SIZE, verbose=1)
        p = predictions[0]

        # Get the class IDs of the detected objects
        class_ids = p['class_ids']

        # Get the indices of the detected buildings
        building_indices = [i for i, class_id in enumerate(class_ids) if class_id == class_names.index("building")]

        results = []

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
            cv2.imwrite(f"./cropimage/hasil/Bulding-{file_name}-{i}.jpg", building)

            image2 = cv2.cvtColor(building, cv2.COLOR_BGR2GRAY)

            # Apply bitwise thresholding to convert the image to black and white
            _, bw_image = cv2.threshold(image2, 100, 255, cv2.THRESH_BINARY)

            n_black_pix = np.sum(bw_image == 0)
            n_white_pix = np.sum(bw_image == 255)
            
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

        def area_in_m2(area_in_pixel,scale):
            return area_in_pixel*scale

        def number_pv_module(area, module_width, module_length):
            return int(area/(module_length*module_width))

        def global_iradiation(location_spesific_solar_irradiance, total_rooftop_area):
            return location_spesific_solar_irradiance*total_rooftop_area

        def maximum_pv_system_capacity(area, module_width, module_length, capacity_per_module):
            pv_number = number_pv_module(area, module_width, module_length)
            # print('pv number:', pv_number)
            return pv_number*capacity_per_module

        def yearly_energy(area, location_spesific_solar_irradiance,efficiency_per_module,system_losses_percentage):
            iradiation = global_iradiation(location_spesific_solar_irradiance, area)
            # print('global iradiation:', iradiation)
            return iradiation*efficiency_per_module*system_losses_percentage
        
        def calculate_potential_energy(area_in_pixel,scale,location_spesific_solar_irradiance,efficiency_per_module,system_losses_percentage):
            area = area_in_m2(area_in_pixel,scale)
            # capacity = maximum_pv_system_capacity(area, module_width, module_length, capacity_per_module)
            # print('Maximum PV system capacity:', capacity)
            energy = yearly_energy(area, location_spesific_solar_irradiance,efficiency_per_module,system_losses_percentage)
            # print('Maximum PV system capacity:', energy)
            return energy

        def calculate_potential_capacity(area_in_pixel,scale, module_width, module_length, capacity_per_module):
            area = area_in_m2(area_in_pixel,scale)
            capacity = maximum_pv_system_capacity(area, module_width, module_length, capacity_per_module)
            return capacity
        
        #TEST
        area_in_pixel = total_rooftop_area 
        scale = scalex
        module_width = widthx
        module_length = lengthx
        capacity_per_module = capacityx
        location_spesific_solar_irradiance = irradiancex
        efficiency_per_module = efficiencyx
        system_losses_percentage = system_lossesx

        total_area = area_in_m2(area_in_pixel, scale)
        max_pv = number_pv_module(area_in_pixel, module_width, module_length)
        global_irradiation = global_iradiation(location_spesific_solar_irradiance, total_rooftop_area)
        energy = calculate_potential_energy(area_in_pixel,scale,location_spesific_solar_irradiance,efficiency_per_module,system_losses_percentage)
        capacity = calculate_potential_capacity(area_in_pixel,scale,module_width, module_length, capacity_per_module)

        return {"data": {"filename": xfilename, "thumbnail": thumbnail_str, "message": "Upload successful!", "data[0]": {"scale":scalex,"module_width":widthx,"module_length":lengthx,"capacity_per_module": capacityx, "location_spesific_solar_irradiance": irradiancex, "efficiency_per_module":efficiencyx, "system_losses_percentage": system_lossesx },  "data[1]": energy, "data[2]": capacity}}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
# GET Image Uploaded
@app.get('/image-result')
async def getImageGeneratedMask():
    global xfilename
    file_path = f"genimage/{xfilename}"
    return FileResponse(file_path)

@app.get('/potential-result')
async def getPotential():
    global energy,capacity,global_irradiation,total_area,max_pv,scale,width,length,capacitypm,irradiance,efficiency,system_losses
    # return {"data": {"potential_energy": energy, "potential_capacity": capacity, "pv_number": max_pv, "global_irradiation": global_irradiation, "total_area": total_area, "userInput": {"scale":scale,"module_width":width,"module_length":length,"capacity_per_module": capacitypm, "location_spesific_solar_irradiance": irradiance, "efficiency_per_module":efficiency, "system_losses_percentage": system_losses }}}
    return {"data": {"potential_energy": energy, "potential_capacity": capacity, "pv_number": max_pv, "global_irradiation": global_irradiation,"total_area": total_area, "userInput": {"scale":scale,"module_width":width,"module_length":length,"capacity_per_module": capacitypm, "location_spesific_solar_irradiance": irradiance, "efficiency_per_module":efficiency, "system_losses_percentage": system_losses }}}
    

@app.get('/potential-resultx')
async def getPotentialx():
    global xfilename,scale,width,length,capacity,irradiance,efficiency,system_losses
    file_name = xfilename

    # Load the image
    image = cv2.imread(os.path.join(IMAGE_DIR, file_name))

    # Detect objects in the image
    predictions = model.detect([image]*config.BATCH_SIZE, verbose=1)
    p = predictions[0]

    # Get the class IDs of the detected objects
    class_ids = p['class_ids']

    # Get the indices of the detected buildings
    building_indices = [i for i, class_id in enumerate(class_ids) if class_id == class_names.index("building")]

    results = []
    total_rooftop_area = 0

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
        cv2.imwrite(f"./cropimage/hasil/Bulding-{file_name}-{i}.jpg", building)

        image2 = cv2.cvtColor(building, cv2.COLOR_BGR2GRAY)

        # Apply bitwise thresholding to convert the image to black and white
        _, bw_image = cv2.threshold(image2, 100, 255, cv2.THRESH_BINARY)

        n_black_pix = np.sum(bw_image == 0)
        n_white_pix = np.sum(bw_image == 255)
        
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
    
    def calculate_potential_energy(area_in_pixel,scale,location_spesific_solar_irradiance,efficiency_per_module,system_losses_percentage):
        area = area_in_m2(area_in_pixel,scale)
        # capacity = maximum_pv_system_capacity(area, module_width, module_length, capacity_per_module)
        # print('Maximum PV system capacity:', capacity)
        energy = yearly_energy(area, location_spesific_solar_irradiance,efficiency_per_module,system_losses_percentage)
        # print('Maximum PV system capacity:', energy)
        return energy

    def calculate_potential_capacity(area_in_pixel,scale, module_width, module_length, capacity_per_module):
        area = area_in_m2(area_in_pixel,scale)
        capacity = maximum_pv_system_capacity(area, module_width, module_length, capacity_per_module)
        return capacity

    #TEST
    area_in_pixel = total_rooftop_area 
    scale = scale
    module_width = width
    module_length = length
    capacity_per_module = capacity
    location_spesific_solar_irradiance = irradiance
    efficiency_per_module = efficiency
    system_losses_percentage = system_losses

    energy_potential_result = calculate_potential_energy(area_in_pixel,scale,location_spesific_solar_irradiance,efficiency_per_module,system_losses_percentage)
    capacity_potential_result = calculate_potential_capacity(area_in_pixel,scale,module_width, module_length, capacity_per_module)

    return {"data": {"area_in_pixel": 45, "scale": scale, "module_width": module_width, "module_length": module_length, "capacity_per_module": capacity_per_module, "location_spesific_solar_irradiance": location_spesific_solar_irradiance, "efficiency_per_module": efficiency_per_module, "system_losses_percentage": system_losses_percentage}}