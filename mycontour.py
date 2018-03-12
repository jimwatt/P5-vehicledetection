import numpy as np
import cv2

im = cv2.imread('test_images/contour.jpg')
imCopy = im.copy()
gray = cv2.bilateralFilter(im, 11, 17, 17)
edged = cv2.Canny(gray, 30, 200)
image, contours, hierarchy =  cv2.findContours(edged,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

print

cv2.drawContours(imCopy,contours[:60],-1,(0,255,0),5)
cv2.imshow('draw contours',imCopy)
cv2.waitKey(0)




ellipse = cv2.fitEllipse(contours[0])
cv2.ellipse(imCopy,ellipse,(0,255,0),8)
cv2.imshow('draw2 contours',imCopy)
cv2.waitKey(0)

cnt = contours[0]
M = cv2.moments(cnt)
print(M)

cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])