import os
from pathlib import Path

"""
This file contains all paths necessary for tensorized_transformers:

An example to import:
from paths import WORKING_DIR, PARENT_WORKING_DIR, DATASET_DIR
"""

# Path of this file and main.py as main working directory:
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
# Parent of main.py to save datasets:
DATASET_DIR = Path(WORKING_DIR).parent
# Path of zip file of the raw dataset:
ZIP_PATH = os.path.join(DATASET_DIR, 'archive.zip')
# Data Folder:
DATA_FOLDER = Path(os.path.join(DATASET_DIR, 'TESS Toronto emotional speech set data'))
