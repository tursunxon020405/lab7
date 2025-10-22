import cv2
import numpy as np

image = cv2.imread('third.jpg')
cv2.imshow('Original', image)

kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

sharpened = cv2.filter2D(image, -2, kernel)

cv2.imshow('Natija', sharpened)

cv2.imwrite('Natija.jpg', sharpened)
#Bu commit
#bu esa 2-commit
cv2.waitKey(0)
cv2.destroyAllWindows()
