#!/usr/bin/env python
# coding: utf-8
#encoding:GBK
# """
# 基于PCA的图像降维及重构
# """
#get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import cv2
import matplotlib.pyplot as plt


# 数据中心化
# 输入参数：矩阵
# 返回值：中心化后的矩阵，扩充至原大小的各列均值矩阵
def Z_centered(dataMat):
    rows,cols=dataMat.shape
    meanVal = np.mean(dataMat, axis=0)  # 按列求均值，即求各个特征的均值
    meanVal = np.tile(meanVal,(rows,1))
    newdata = dataMat-meanVal
    return newdata, meanVal
 

# 求协方差矩阵
def Cov(dataMat):
    meanVal = np.mean(dataMat,0) #压缩行，返回1*cols矩阵，对各列求均值
    rows=dataMat.shape[0]
    meanVal = np.tile(meanVal, (rows,1)) #把数组沿各个方向复制(y,x)，返回rows行的均值矩阵
    Z = dataMat - meanVal
    Zcov = (1/(rows-1))*Z.T * Z#协方差公式
    return Zcov
    
#最小化降维造成的损失，确定k
def Percentage2n(eigVals, percentage):
    sortArray = np.sort(eigVals)  # 升序
    sortArray = sortArray[-1::-1]  # 逆转，即降序
    arraySum = sum(sortArray)
    tmpSum = 0
    num = 0
    for i in sortArray:
        tmpSum += i
        num += 1
        if tmpSum >= arraySum * percentage:
            return num
    
#得到最大的k个特征值和特征向量
def EigDV(covMat, p):
    D, V = np.linalg.eig(covMat) # 得到特征值和特征向量covMat=V^-1DV
    k = Percentage2n(D, p) # 确定k值
    print("保留99%信息，降维后的特征个数："+str(k)+"\n")
    print("D ",D.shape," V ",V.shape)
    
    eigenvalue = np.argsort(D)#将矩阵D按照axis排序，并返回排序后的下标
    K_eigenValue = eigenvalue[-1:-(k+1):-1]
    K_eigenVector = V[:,K_eigenValue]#行全部选取，列选取特征值对应的下标列
    return D[0:k], K_eigenVector
    
#得到降维后的数据
def getlowDataMat(DataMat, K_eigenVector):
    return DataMat * K_eigenVector
 
#重构数据
def Reconstruction(lowDataMat, K_eigenVector, meanVal):
    reconDataMat = lowDataMat * K_eigenVector.T + meanVal
    return reconDataMat
 
# """
# PCA算法
# 功能：降维至保留p的信息
# 传入参数：data图片，p保留百分比
# 返回值：
# """
def PCA(data, p):
    dataMat = np.float32(np.mat(data))
    #数据中心化
    dataMat, meanVal = Z_centered(dataMat)
    #计算协方差矩阵
    covMat = Cov(dataMat)#covMat[i][j]表示第i，j列的相关度
    #covMat = np.cov(dataMat, rowvar=0)
    #得到最大的k个特征值和特征向量
    D, V = EigDV(covMat, p)#(40,)   (256, 40)
    print(D,V)
    #得到降维后的数据
    lowDataMat = getlowDataMat(dataMat, V)#(256,256)*(256,40)=(256, 40)
    #[D]^-1V[D]=covMat
    #lowDateMate=dateMate*V
    #重构数据
    reconDataMat = Reconstruction(lowDataMat, V, meanVal)
    #re=low*V+meanVal
    return reconDataMat

def main():
    imagePath = '105.png'
    image = cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE)
    #image=[[1,2,3],[3,5,6],[3,8,9]]
    #image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # cv2.imshow("image",image)
    
    # cv2.waitKey(0) #等待按键
    canny=cv2.Canny(image,35,200)
    rows,cols=image.shape
    print("降维前的特征个数："+str(cols)+"\n")

    reconImage = PCA(image, 0.99)
    reconImage = reconImage.astype(np.uint8)
    canny_PCA = cv2.Canny(reconImage, 35, 200)
    
    plt.figure(figsize=(20,20))
    pic_row,pic_col=2,2
    plt.axis('off')
    
    plt.subplot(pic_row,pic_col,1)
    plt.imshow(image,cmap='gray')
    plt.title("origin")
    plt.subplot(pic_row,pic_col,2)
    plt.imshow(reconImage,cmap='gray')
    plt.title("recon_PCA")
    plt.subplot(pic_row,pic_col,3)
    plt.title("without PCA canny")
    plt.imshow(canny,cmap='gray')
    plt.subplot(pic_row,pic_col,4)
    plt.imshow(canny_PCA,cmap='gray')
    plt.title("PCA-canny")
    plt.show()
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
if __name__=='__main__':
    main()
 



