
import cv2
import matplotlib.pyplot as plt
import numpy as np

def CannyThreshold(lowThreshold):
    detected_edges = cv2.GaussianBlur(img,(3,3),0)
    detected_edges = cv2.Canny(detected_edges,lowThreshold,lowThreshold*ratio,apertureSize = kernel_size)
    dst = cv2.bitwise_and(img,img,mask = detected_edges)  # just add some colours to edges from original image.
    cv2.imshow('Adaptive Canny',dst)
 
lowThreshold = 0
max_lowThreshold = 100
ratio = 3
kernel_size = 3
 
img = cv2.imread('C:\\Users\\25921\\Desktop\\test.jpg')
 
cv2.namedWindow('Adaptive Canny')
 
cv2.createTrackbar('Min thre','Adaptive Canny',lowThreshold, max_lowThreshold, CannyThreshold)
 
CannyThreshold(0)  
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()


