import cv2 as cv
import numpy as np


image = cv.imread("images/img.jpeg")
cv.imshow("Original_image", image)

filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpedned = cv.filter2D(image, -1, filter)
cv.imshow("sharp image", sharpedned)

cv.waitKey(0)
cv.destroyAllWindows()
