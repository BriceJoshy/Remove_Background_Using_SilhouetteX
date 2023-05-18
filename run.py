# image segmentation is the task of labelling each pixel of an image with a specific class
# background:0
# foreground:1
"""Using the deeplabv3+ model silhoutteX.h5"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2 as cv

#  extract all the input images
from glob import glob

# progress bar
from tqdm import tqdm

# ml library
import tensorflow as tf
from tensorflow import keras

# Code within a with statement will be able to access custom objects by name
from keras.utils import CustomObjectScope
from metrics import dice_loss, dice_coeff, iou

""" Defining Global Parameters """
# height
H = 512
# width
W = 512


""" Function for creating the directory """


#  i.e is a blank directory
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


#  to get reproducible results all the time
""" Seeding the environment """

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    """Creating the directory for storing the files"""
    create_directory("remove_background")

    """Loading the model S_X"""

    with CustomObjectScope(
        {"iou": iou, "dice_coef": dice_coeff, "dice_loss": dice_loss}
    ):
        model = tf.keras.models.load_model("silhouetteX.h5")
