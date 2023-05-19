
    # """Reading the images"""
    # # always convert image to the 3 channel BGR color image.
    # image = cv.imread(path, cv.IMREAD_COLOR)
    # # print(image.shape)
    # #  now the image is an numpy array
    # # i.e why the _ is given as there are more parameters in shape
    # # save for later
    # height, width, _ = image.shape
    # print(image.shape)
    # resized_image = cv.resize(image, (W, H))
    # # print(resized_image.shape)
    # # Normalization of the resized_image
    # # now the range of the pixel value is btw 0 and 1 as is divided by the max pixel value
    # resized_image = resized_image / 255.0
    # resized_image = resized_image.astype(np.float32)
    # # adding th expanded dimension to the axis
    # resized_image = np.expand_dims(resized_image, axis=0)
    # #  (1,512,512,3) this is because this is a single image and we give the numpy array into the model by batches
    # #  so we are giving it as one batch
    # # print(resized_image.shape)

    # """Making a Prediction"""
    # # because we have given one image as an input
    # # ouput would also be one mask not more than that will be predicted by the model
    # predicted_mask = model.predict(resized_image)[0]
    # # index 0 so the shape became as below
    # # (1, 512, 512, 1) as it is an binary mask
    # # print(predicted_image.shape)
    # #  this is the mask
    # #  we resize the prdicted mask is resized to 512,512
    # #  we make sure that the the size of the prediction mask is same as the origianl image
    # predicted_mask = cv.resize(predicted_mask, (width, height))
    # # print(predicted_image.shape)
    # #  now we are going to expand its dimentions for the last axis
    # predicted_mask = np.expand_dims(predicted_mask, axis=-1)
    # # print(predicted_image.shape)

    # # creaing the threshold
    # predicted_mask = predicted_mask > 0.5
    # # predicted maks contains range of 0 and 1
    # # multiplying it by 255 to get the range of 0 to 255
    # """saving the predicted mask"""
    # # cv.imwrite(f"remove_background/{image_name}.png", predicted_mask * 255)

    # # Need two kinds of masks
    # """Photo mask"""
    # # photo_mask is the main object
    # photo_mask = predicted_mask
    # background_mask = np.abs(1 - predicted_mask)
    # # print(background_mask.shape)
    # # cv.imwrite(f"remove_background/{image_name}.png", background_mask * 255)
    # # reversed the color
    # #  photot mask contain the mask for the main object and the background mask contain the mask for the background
    # """why not * 255"""
    # # we cant see the difference btw the 0 and 1 but we can see the diff os the 0 and 255
    # # here by multiplying - because the background ,ask or the photo mask is for the array contain the values 0 or 1
    # # pixel values multiplied by zero became 0 and the others are multiplied by  so the object remain the same in the result

    # # cv.imwrite(f"remove_background/{image_name}.png", image * photo_mask)
    # # seperating the background removing the object
    # # cv.imwrite(f"remove_background/{image_name}.png", image * background_mask)

    # """Custom background (color)"""

    # masked_image = image * photo_mask
    # #  to have a custom color there is need of 3 color channels
    # #  checking the channel from shape
    # # print(masked_image.shape)

    # # stacking the 3 values of top of each other in the last axis
    # background_mask = np.concatenate(
    #     [background_mask, background_mask, background_mask], axis=-1
    # )
    # background_mask = background_mask * [255, 255, 255]
    # final_image = masked_image + background_mask
    # cv.imwrite(f"removed_background_image/{image_name}.png", final_image)
