# -*- coding: utf-8 -*-
import numpy as np
import argparse
import glob
import cv2
import os

# 自适应阈值的Canny函数
def auto_canny(image, sigma=0.33):
	# 计算单通道像素强度的中位数
	v = np.median(image)

	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	return edged

# Gamma变换增强边缘对比
def adjust_gamma(image, gamma):
    invGamma = 1.0/gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255)
    table = np.array(table).astype("uint8")
    return cv2.LUT(image, table)


img=cv2.imread('C:\\Users\\25921\\Desktop\\test.jpg')
cv2.imshow('Origin',img)

img= adjust_gamma(img,2)
#高斯滤波与中值滤波
blurred = cv2.medianBlur(img,3,img)
#blurred = cv2.GaussianBlur(img, (3, 3), 0)

low = cv2.Canny(blurred, 20, 80)
high = cv2.Canny(blurred, 50, 100)
auto = auto_canny(blurred)

showall=np.hstack((low,high,auto))
cv2.imshow('LOW / HIGH / AUTO',showall)
   
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
