import cv2 as cv
from cv2 import LINE_AA
import numpy as np

image = cv.imread("removed_background_images/XRVOS_myn_10653722_1100x1100.png")
cv.imshow("Originalimage", image)

grey_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# cv.imshow("Grey image", grey_image)

ret, threshold = cv.threshold(grey_image, 1, 255, cv.THRESH_OTSU)
# cv.imshow("Threshold image", threshold)

edge = cv.Canny(threshold, 400, 400)
cv.imshow("edges", edge)

dialated = cv.dilate(edge, (1, 1), iterations=3)
cv.imshow("Dialated image", dialated)
contours, heirarchy = cv.findContours(dialated, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(str(len(contours)))
x, y, w, h = cv.boundingRect(contours[-1])
# for contour in range(len(contours)):
#     #  draw current contours
#     cv.drawContours(image, contours, contour, (0, 255, 0), 3)
#     cv.imshow("Rectangle", image)
#     cv.waitKey(0)

rectangle = cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

cv.imshow("Rectangle", image)
cropped_image = image[y : y + h, x : x + w]
cv.imshow("cropped", cropped_image)


cv.waitKey(0)
cv.destroyAllWindows()
