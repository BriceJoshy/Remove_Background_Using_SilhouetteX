import cv2 as cv
import numpy as np
from PIL import Image
from PIL import ImageFilter


image = Image.open("images/img.jpeg")
image.show()

filter = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
filter.show()

ret, threshold = cv.threshold(filter, 1, 255, cv.THRESH_OTSU)
cv.imshow("Threshold image", threshold)

edges = filter.filter(ImageFilter.FIND_EDGES)
edges.show()

cv.waitKey(0)
cv.destroyAllWindows()
