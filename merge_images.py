import cv2
import sys

background = cv2.imread("harry.jpg")
face = cv2.imread("roi.png")

added_image = cv2.addWeighted(background,0.4,face,0.1,0)

cv2.imshow("Merged", added_image);