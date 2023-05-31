import cv2 as cv
import numpy as np
from PIL import Image
from PIL import ImageFilter


image = Image.open("images/img.jpeg")
image.show()

filter = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
filter.show()

edges = filter.filter(ImageFilter.UnsharpMask)
edges.show()

cv.waitKey(0)
cv.destroyAllWindows()
