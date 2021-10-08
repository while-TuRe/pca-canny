# coding=gbk
import cv2
import matplotlib.pyplot as plt
import numpy as np

img=cv2.imread('C:\\Users\\25921\\Desktop\\test.jpg')
cv2.imshow('Origin',img)

def adjust_gamma(image, gamma):
    invGamma = 1.0/gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255)
    table = np.array(table).astype("uint8")
    return cv2.LUT(image, table)

#Gamma变换
img= adjust_gamma(img,0.5)

#高斯滤波与中值滤波
#img = cv2.GaussianBlur(img,(3,3),0)
img = cv2.medianBlur(img,3,img)

case1=cv2.Canny(img,20,80)
case2=cv2.Canny(img,50,100)

showboth=np.hstack((case1,case2))

cv2.imshow('LOW / HIGH',showboth)

if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()