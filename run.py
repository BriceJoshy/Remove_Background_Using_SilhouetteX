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
    create_directory("removed_background_images")
    create_directory("cropped_images")

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
    # print(image_data)
    new_data = glob("removed_background_images\*")

    for path in tqdm(image_data, total=len(image_data)):
        """extracting the name of the images"""
        image_name = path.split("\\")[-1].split(".")[0]
        print(image_name)

        """Reading the images"""
        # always convert image to the 3 channel BGR color image.
        image = cv.imread(path, cv.IMREAD_COLOR)
        # print(image.shape)
        #  now the image is an numpy array
        # i.e why the _ is given as there are more parameters in shape
        # save for later
        height, width, _ = image.shape
        resized_image = cv.resize(image, (W, H))
        # print(resized_image.shape)
        # Normalization of the resized_image
        # now the range of the pixel value is btw 0 and 1 as is divided by the max pixel value
        resized_image = resized_image / 255.0
        # print(resized_image.shape)
        """Explanation for the type conversion"""
        #  https://stackoverflow.com/questions/59986353/why-do-i-have-to-convert-uint8-into-float32
        resized_image = resized_image.astype(np.float32)
        # print(resized_image.shape)
        # adding th expanded dimension to the axis
        resized_image = np.expand_dims(resized_image, axis=0)
        #  (1,512,512,3) this is because this is a single image and we give the numpy array into the model by batches
        #  so we are giving it as one batch
        # print(resized_image.shape)

        """Making a Prediction"""
        # because we have given one image as an input
        # ouput would also be one mask not more than that will be predicted by the model
        predicted_mask = model.predict(resized_image)[0]
        # index 0 so the shape be came as below
        # (1, 512, 512, 1) as it is an binary mask
        # print(predicted_image.shape)
        #  this is the mask
        #  we resize the prdicted mask is resized to 512,512
        #  we make sure that the the size of the prediction mask is same as the origianl image
        predicted_mask = cv.resize(predicted_mask, (width, height))
        print(predicted_mask.shape)
        #  now we are going to expand its dimentions for the last axis
        predicted_mask = np.expand_dims(predicted_mask, axis=-1)
        print(predicted_mask.shape)

        # creaing the threshold
        predicted_mask = predicted_mask > 0.5
        # predicted maks contains range of 0 and 1
        # multiplying it by 255 to get the range of 0 to 255
        """saving the predicted mask"""
        # cv.imwrite(f"remove_background/{image_name}.png", predicted_mask * 255)

        # Need two kinds of masks
        """Photo mask"""
        # photo_mask is the main object
        photo_mask = predicted_mask
        background_mask = np.abs(1 - predicted_mask)
        # print(background_mask.shape)
        # cv.imwrite(f"remove_background/{image_name}.png", background_mask * 255)
        # reversed the color
        #  photot mask contain the mask for the main object and the background mask contain the mask for the background
        """why not * 255"""
        # we cant see the difference btw the 0 and 1 but we can see the diff os the 0 and 255
        # here by multiplying - because the background ,ask or the photo mask is for the array contain the values 0 or 1
        # pixel values multiplied by zero became 0 and the others are multiplied by  so the object remain the same in the result

        # cv.imwrite(f"remove_background/{image_name}.png", image * photo_mask)
        # seperating the background removing the object
        # cv.imwrite(f"remove_background/{image_name}.png", image * background_mask)

        """Custom background (color)"""

        masked_image = image * photo_mask
        #  to have a custom color there is need of 3 color channels
        #  checking the channel from shape
        # print(masked_image.shape)

        # stacking the 3 values of top of each other in the last axis
        background_mask = np.concatenate(
            [background_mask, background_mask, background_mask], axis=-1
        )
        background_mask = background_mask * [255, 255, 255]
        final_image = masked_image + background_mask
        cv.imwrite(f"removed_background_images/{image_name}.png", final_image)

        """Cropping the image from drawn contour"""

    for path in tqdm(new_data, total=len(new_data)):
        copy_image = cv.imread(path)

        grey_image = cv.cvtColor(copy_image, cv.COLOR_BGR2GRAY)
        # cv.imshow("Grey image", grey_image)

        ret, threshold = cv.threshold(grey_image, 1, 255, cv.THRESH_OTSU)
        # cv.imshow("Threshold image", threshold)

        edge = cv.Canny(threshold, 400, 400)
        # cv.imshow("edges", edge)

        dialated = cv.dilate(edge, (1, 1), iterations=3)
        # cv.imshow("Dialated image", dialated)
        contours, heirarchy = cv.findContours(
            dialated, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE
        )
        print(str(len(contours)))
        x, y, w, h = cv.boundingRect(contours[-1])
        # for contour in range(len(contours)):
        #     #  draw current contours
        #     cv.drawContours(image, contours, contour, (0, 255, 0), 3)
        #     cv.imshow("Rectangle", image)
        #     cv.waitKey(0)

        rectangle = cv.rectangle(copy_image, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # cv.imshow("Rectangle", image)
        cropped_image = copy_image[y : y + h, x : x + w]
        cv.imwrite(f"cropped_images/{image_name}.png", cropped_image)
