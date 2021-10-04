import cv2 as cv
import numpy as np
from math import *
def main():
    img_truth = cv.imread('1-truth.jpg')#BGR
    img = cv.imread('1-truth.jpg')#BGR
    # img = salt_pepper_noise(img,0.01)
    # cv.imwrite('1-noise.jpg',img)
    if(img is None or img_truth is None):
        print("图片读取失败")
        return 
    TP=0
    TN=0
    FP=0
    FN=0
    row,col,n=img.shape
    for i in range(row):
        for j in range(col):
            #FP:本来不是灰质，判断成灰
            if img_truth[(i,j,0)]==max(img_truth[(i,j)]):#G<B,灰质，P
                if img[(i,j,0)]==max(img[(i,j)]):#一致，T
                    TP=TP+1
                else:
                    FP=FP+1
            else:
                if img[(i,j,0)]!=max(img[(i,j)]):
                    TN=TN+1
                else:
                    FN=FN+1
    print(TP,TN,FP,FN)
    AC=(TP+TN)/(TP+TN+FP+FN)
    SE=TP/(TP+FN)
    SP=TN/(FP+TN)
    DI=2*TP/(2*TP+FP+FN)
    JA=TP/(TP+FP+FN)
    print("准确率",AC)
    print("敏感度(灰质中被正确灰质的比率)",SE)
    print("特异性(非灰质中被正确判断为非灰质的比率)",SP)
    print("骰子系数",DI)
    print("Jaccard指数(交并比，关注部分的AC)",JA)

#分类，背景是0，灰质是1，白质是2
def caseImg(img):#BGR
    row,col,n=img.shape
    claImg=np.zeros((row,col))
    for i in range(row):
        for j in range(col):
            if img[(i,j,0)]==max(img[(i,j)]) and img[(i,j,0)]>50:
                claImg[(i,j)]=1
            elif img[(i,j,1)]==max(img[(i,j)]) and img[(i,j,1)]>50:
                claImg[(i,j)]=2
    return claImg
def salt_pepper_noise(image, prob):
    """
    添加椒盐噪声
    :param image: 输入图像
    :param prob: 噪声比
    :return: 带有椒盐噪声的图像
    """
    salt = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.rand()
            if rdn < prob:
                salt[i][j] = [255,0,0]
            elif rdn > thres:
                salt[i][j] = [0,255,0]
            else:
                salt[i][j] = image[i][j]
    return salt

def structures():
    img = cv.imread('1-result.jpg')#BGR
    img = salt_pepper_noise(img,0.01)
    cv.imwrite('1-noise.jpg',img)
    if(img is None):
        print("图片读取失败")
        return 
    img = caseImg(img)
    row,col=img.shape
    A=0
    B=0
    dirct=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]
    for i in range(row):
        for j in range(col):
            n=0
            for d in dirct:
                pos=(i+d[0],j+d[1])
                if(pos[0]>=0 and pos[0]<row and pos[1]>=0 and pos[1]<col):
                    if img[pos]==img[(i,j)]:
                        n=n+1
            if n<=3:
                A=A+1
            if img[(i,j)]!=0:
                B=B+1
    print("SRN=",10*log(B/A,10))

if __name__=='__main__':
    main()
    structures()