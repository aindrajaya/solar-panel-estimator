# GIZ
Solar panel calculator (GIZ x PENA)

## Prerequisites
1. Using **Python 3.6.15** or any Python version **3.6** (Recommended, to avoid error)
2. Install Conda (if needed, to make virtual environment)
## How to Run
1. Install all the dependencies needed
`$ pip install -r requirements.txt`
2. Install Pycocotools
*Windows
`$ pip install pycocotools-windows`
Mac
`$ pip install pycocotools`
3. Install Python FastAPI, click [here](https://realpython.com/fastapi-python-web-apis/) to get more informatio
```bash
$ python -m pip install fastapi uvicorn[standard]
```
4. Open `notebooks` folder and setup MRCNN libraries
```bash
$ cd utils
$ python setup.py install
```
5. Create `uploads`, `genimage` and `cropimage` fodler on the root/home folder
6. Run FastAPI python
```bash
$ uvicorn app.main:app --reload
```
7. Make sure that libs from folder `notebooks` on the right path