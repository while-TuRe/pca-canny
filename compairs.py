import numpy as np
import cv2 as cv
import os
from PIL import Image
def show(cv_img):
    """
    传入cv类型，转为PIL类型显示
    """
    # OpenCV转换成PIL.Image格式
    image = Image.fromarray(cv.cvtColor(cv_img, cv.COLOR_BGR2RGB))
    image.show()
def printImg(img):
    row,col=img.shape
    for i in range(row):
        for j in range(col):
            print(img[i][j],end=" ")
        print("")

# 读取图像
img = cv.imread('072.png', cv.COLOR_BGR2GRAY)
#算子
Robertsx=np.array([[-1,0],[0,1]],dtype=int)
Robertsy=np.array([[0,-1],[1,0]],dtype=int)
Prewittx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]],dtype=int)
Prewitty = np.array([[-1,0,1],[-1,0,1],[-1,0,1]],dtype=int)
Sobelx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=int)
Sobely = np.array([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=int)
Laplacian4 = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]],dtype=int)
Laplacian8 = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],dtype=int)
log = np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]],dtype=int)
kernels=[(Robertsx,Robertsy,"Roberts"), (Prewittx,Prewitty,"Prewitt"),
 (Sobelx,Sobely,"Sobel"),
 (Laplacian4,Laplacian4,"Laplacian4"),(Laplacian8,Laplacian8,"Laplacian8"),
 (log,log,"log")]
#卷积计算
for (kernelx,kernely,name)in kernels:
    x = cv.filter2D(img, cv.CV_8U, kernelx)
    y = cv.filter2D(img, cv.CV_8U, kernely)
    # 转 uint8 ,图像融合
    absX = cv.convertScaleAbs(x)
    absY = cv.convertScaleAbs(y)
    result = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
    #show(result)
    cv.imwrite(name+'.png',result)
