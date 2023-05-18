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

    """Summary of the model"""
    # from the summary we can see that the imput is a 512,512,3 channel image
    # and the output is a 512,512,1 channel (binary mask)
    # model.summary()

    """loading the images/dataset to the model"""
    image_data = glob("images\*")
    print(image_data)

    for path in tqdm(image_data, total=len(image_data)):
        """extracting the name of the images"""
        image_name = path.split("\\")[-1].split(".")[0]
        print(image_name)

    """Reading the images"""
    # always convert image to the 3 channel BGR color image.
    image = cv.imread(path, cv.IMREAD_COLOR)
    print(image.shape)
    #  now the image is an numpy array
    # i.e why the _ is given as there are more parameters in shape
    # save for later
    height, width, _ = image.shape
    resized_image = cv.resize(image, (H, W))
    print(resized_image.shape)
