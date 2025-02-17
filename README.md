# GIZ
Solar panel calculator
This FastAPI application processes uploaded images to detect buildings and calculate potential solar energy generation. It utilizes the Mask R-CNN model for object detection to identify buildings, computes rooftop areas, and estimates solar energy capacity and generation potential.

**Docs and API Docs: (TBA)**

## Key Features
- Image processing using OpenCV, PIL, and Mask R-CNN.
- Calculation of rooftop areas, solar energy potential, and PV system capacity.
- RESTful API endpoints for uploading images and retrieving results.

## Prerequisites
1. Using **Python 3.6.15** or any Python version **3.6** (Recommended, to avoid error)
2. Install Conda (if needed, to make virtual environment)

## How to Run
1. Install all the dependencies needed
    `$ pip install -r requirements.txt`
2. Install Pycocotools
    * Windows
        `$ pip install pycocotools-windows`
    * Mac
        `$ pip install pycocotools`
3. Install Python FastAPI. For more information, click [here](https://realpython.com/fastapi-python-web-apis/):
    ```bash
    $ python -m pip install fastapi uvicorn[standard]
    ```
4. Set up MRCNN libraries:
    ```bash
    $ notebooks/utils
    $ python setup.py install
    ```
5. Create the following folders in the root/home directory:
    -  `uploads`: For storing uploaded images.
    - `genimage`: For saving processed images.
    - `cropimage`: For storing cropped images.
6. Run FastAPI python
    ```bash
    $ uvicorn app.main:app --reload
    ```
7. Ensure that the libraries from the `notebooks` folder are in the correct path.

## API Endpoints
- GET `/`: Returns a welcome message.
- POST `/uploadimage`: Handles image uploads, processes images using Mask R-CNN to detect buildings, and calculates rooftop areas, solar energy, and capacity.
- GET `/image-result`: Returns the processed image with detected buildings.
- GET `/potential-result`: Returns the calculated solar energy and capacity.

## Helper Functions
The application includes several helper functions to:
- Calculate rooftop area.
- Estimate the number of PV modules.
- Compute global irradiation.
- Determine PV system capacity.
- Calculate yearly energy potential.

For detailed implementation, refer to the code in the repository.