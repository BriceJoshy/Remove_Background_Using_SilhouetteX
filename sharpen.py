import cv2 as cv
import numpy as np


image = cv.imread("images/upscale.jpg")
cv.imshow("Original_image", image)

filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpedned = cv.filter2D(image, -1, filter)
cv.imshow("sharp image", sharpedned)

cv.waitKey(0)
cv.destroyAllWindows()
