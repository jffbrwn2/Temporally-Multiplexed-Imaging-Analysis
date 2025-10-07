#!/usr/bin/env python3
"""
Example configuration file for fluorescence unmixing.

Copy this file and modify the paths to match your data structure.
"""

import os

# Base directory containing your data
BASE_PATH = "/path/to/your/data"

# Path to reference curve data (datRef.csv or datRefK.csv)
# The script will automatically select the correct file based on --fitted flag
REF_PATH = os.path.join(BASE_PATH, "datRef.csv")

# Folder containing green fluorescence curve files (*green_curve_2.mat)
GREEN_FOLDER = os.path.join(BASE_PATH, "green_mat_curves")

# Folder containing red mask files (*red_mask.mat)
RED_MASK_FOLDER = os.path.join(BASE_PATH, "red_masks")

# Output directory for results
OUTPUT_DIR = os.path.join(BASE_PATH, "unmixing_output")

# Image dimensions (pixels)
IMAGE_WIDTH = 1360
IMAGE_HEIGHT = 1360

# Number of fluorescence channels
NUM_COLORS = 5

# Example file naming convention:
# Green curves: A1_ROI_1_green_curve_2.mat, A1_ROI_2_green_curve_2.mat, ...
# Red masks:    A1_ROI_1_red_mask.mat,       A1_ROI_2_red_mask.mat, ...
