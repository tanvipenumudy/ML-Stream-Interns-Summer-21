import cv2 

img = cv2.imread('img1.png')
gray=cv2.imread('img1.png',cv2.IMREAD_GRAYSCALE)

cv2.imshow('Icon', img)
cv2.imshow('Gray ICon',gray)

cv2.waitKey(0)
cv2.destroyAllWindows()