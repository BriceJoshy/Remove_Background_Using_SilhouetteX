filter = np.array([[-1, -1, -1], [-1, 9, -1], [0, -1, 0]])
sharpedned = cv.filter2D(image, -1, filter)
cv.imshow("sharp image", sharpedned)