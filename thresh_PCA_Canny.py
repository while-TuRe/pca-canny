import numpy as np
import pywt
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
from skimage import data, filters
from numpy import *
import numpy as np
from scipy import signal


def threshSym4(imgName):
    # 1  读入图片数据
    img = cv.imread(imgName)

    # 将多通道图像变为单通道图像
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.float32)
    noiseSigma = 0.04
    img = img + random.normal(0, noiseSigma, size=img.shape)  # 生成正态分布

    # 2分解一层
    coeffs = pywt.wavedec2(img, 'sym4', level=1)
    cA, (cH, cV, cD) = coeffs

    # 将各个子图进行拼接，最后得到一张图（影像压缩）
    AH = np.concatenate([cA, cH], axis=1)  # np.concatenate 数组拼接函数 axis表示对应行
    VD = np.concatenate([cV, cD], axis=1)
    img1 = np.concatenate([AH, VD], axis=0)

    ##可以进行图像阈值分割/（影像分割）
    thresh = filters.threshold_otsu(img)  # 返回一个阈值
    dst = (img <= thresh) * 1.0  # 根据阈值进行分割

    # 软阈值 重构  计算系数
    # pywt.thresholding.软，尤其是它的lambda表达式：lambda x: pywt.thresholding.soft(x, threshold)

    # threshold为误差所带来的影响
    threshold = noiseSigma * sqrt(2 * log2(img.size))
    NewImage = pywt.waverec2(coeffs, 'haar')

    titles = ['Source', 'compression', 'segmentation', "reconstruction"]
    images = [img, img1, dst, NewImage]
    for i in range(4):
        plt.subplot(1, 4, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
    plt.savefig("reconstruction.png")
    return thresh


def dst(imgName, thresh):
    ########## open the image ##########
    # im=Image.open(imgName)
    im = cv.imread(imgName)
    # rgb_im = im.convert('RGB')  # convert RGBA to RGB
    rgb_im = cv.cvtColor(im, cv.COLOR_BGR2RGB).astype(np.float32)
    # it turns out the pics aren't segmented well,they just cover the skull and the eye balls
    ########## open the image ##########
    for i in range(rgb_im.shape[0]):
        for j in range(rgb_im.shape[1]):
            # r, g, b = rgb_im.getpixel((i, j))
            r, g, b = rgb_im[i][j]
            if r <= thresh:
                # rgb_im.putpixel((i, j), (0, 0, 0))
                rgb_im[i][j] = (0, 0, 0)
    plt.imshow(rgb_im)
    plt.savefig("073.png")
    ########## draw histogram ##########
    # r, g, b = rgb_im.split()  # split R/G/B
    r, g, b = cv.split(rgb_im)
    plt.figure("flowerHist")
    ar = np.array(r).flatten()  # flatten numpy into 1D_array,折叠称直方图
    plt.hist(ar, bins=256, density=True, facecolor="green", edgecolor="green", alpha=0.5)
    plt.savefig("Hist.png")
    ########## draw histogram ##########

    ########## manually count the number of pixels of every grayscale from 0 to 255 ##########
    xArray = range(0, 256)  # grayscale from 0 to 255
    countArray = [0 for i in range(0, 256)]  # countArray to record the number of pixels in different grayscale
    pixelNum = len(ar)
    for i in range(0, 256):
        for j in range(0, pixelNum):
            if (ar[j] == i):
                countArray[i] += 1
    # 统计各灰度值有多少个灰度点
    countArray[0] = 0  # set the number of pixel_0 as 0 to remove its effect on the outcome
    ########## manually count the number of pixels of every grayscale from 0 to 255 ##########

    ########## 15-polynomial fitting  ##########
    fit = np.polyfit(xArray, countArray, 15)
    factor = np.poly1d(fit)
    yPoly = factor(xArray)  # polynomial
    ########## to draw the histogram and the fitted polynomial  ##########
    plt.plot(xArray, yPoly, color="r", label='polyfit values')
    plt.savefig("polyfit.png")
    plt.show()
    ########## to draw the histogram and the fitted polynomial  ##########

    ########## to find the appropriate threshold to seg  ##########
    peaks_lower = signal.argrelextrema(yPoly, np.less)[0]  # minimum numpy
    peaks = signal.find_peaks(yPoly, distance=10)  # maximum numpy
    peakList_Index = peaks[0].tolist()  # turn maximum numpy into list

    print("The minimum list of the grayscale is as below")
    print(peaks_lower)
    print("The maximum list of the grayscale is as below")
    print(peakList_Index)

    peakList = [0 for i in range(0, len(peakList_Index))]
    for i in range(0, len(peakList_Index)):
        peakList[i] = yPoly[peakList_Index[i]]

    ####figure out the top 2 grayscale
    peak_1 = max(peakList)
    peakIndex_1 = peakList.index(peak_1)

    peakList[peakIndex_1] = 0

    peak_2 = max(peakList)
    peakList[peakIndex_1] = peak_1
    peakIndex_2 = peakList.index(peak_2)

    lowerLimit = peakList_Index[peakIndex_2]
    higherlimit = peakList_Index[peakIndex_1]
    ####figure out the top 2 grayscale

    threshold_tmp = 0
    threshold = peaks_lower[0]
    for i in range(0, len(peaks_lower)):
        if peaks_lower[i] > lowerLimit and peaks_lower[i] < higherlimit:
            threshold_tmp = peaks_lower[i]
            if threshold < lowerLimit or threshold > higherlimit or yPoly[threshold] >= yPoly[threshold_tmp]:
                threshold = threshold_tmp

    print("the threshold turns out to be:")
    print(threshold)
    ########## to find the appropriate threshold to seg  ##########

    ########## use the threshold to seg  ##########
    for i in range(rgb_im.shape[0]):
        for j in range(rgb_im.shape[1]):
            # r, g, b = rgb_im.getpixel((i, j))
            r, g, b = rgb_im[i][j]
            # set whitematter as green
            if r >= threshold:
                # rgb_im.putpixel((i, j), (255, 0, 0))
                rgb_im[i][j] = (255, 0, 0)
    ########## use the threshold to seg  ##########

    ########## show the outcome  ##########
    plt.imshow(rgb_im)
    plt.savefig("dst1.png")
    plt.show()
    return rgb_im
    # rgb_im.show()
    ########## show the outcome  ##########


def Canny(dstImg):
    # img = cv.imread("dst1.png")
    #
    # # 将多通道图像变为单通道图像
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.float32)
    sigma1 = 1
    sigma2 = 1
    sum = 0

    def BGR2GRAY(img):
        b = img[:, :, 0].copy()
        g = img[:, :, 1].copy()
        r = img[:, :, 2].copy()

        # Gray scale
        out = 0.2126 * r + 0.7152 * g + 0.0722 * b
        out = out.astype(np.uint8)

        return out

    gaussian = np.zeros([5, 5])
    for i in range(5):
        for j in range(5):
            gaussian[i, j] = math.exp(-1 / 2 * (np.square(i - 3) / np.square(sigma1)  # 生成二维高斯分布矩阵
                                                + (np.square(j - 3) / np.square(sigma2)))) / (
                                     2 * math.pi * sigma1 * sigma2)
            sum = sum + gaussian[i, j]
    gaussian = gaussian / sum

    # print(gaussian)

    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    # step1.高斯滤波
    # gray = rgb2gray(img)
    gray = BGR2GRAY(dstImg)
    # print("gray")
    # print(gray.shape)
    # print("dstImg")
    # print(dstImg.shape)
    W, H = gray.shape
    # W,H-5
    new_gray = np.zeros([W, H])
    for i in range(W - 5):
        for j in range(H - 5):
            new_gray[i, j] = np.sum(gray[i:i + 5, j:j + 5] * gaussian)  # 与高斯矩阵卷积实现滤波

    # plt.imshow(new_gray, cmap="gray")

    # step2.增强 通过求梯度幅值
    W1, H1 = new_gray.shape
    # w1,h1-1
    dx = np.zeros([W1 - 1, H1 - 1])
    dy = np.zeros([W1 - 1, H1 - 1])
    d = np.zeros([W1, H1])
    for i in range(W1 - 1):
        for j in range(H1 - 1):
            dx[i, j] = new_gray[i, j + 1] - new_gray[i, j]
            dy[i, j] = new_gray[i + 1, j] - new_gray[i, j]
            d[i, j] = np.sqrt(np.square(dx[i, j]) + np.square(dy[i, j]))  # 图像梯度幅值作为图像强度值

    # plt.imshow(d, cmap="gray")

    # setp3.非极大值抑制 NMS
    W2, H2 = d.shape
    NMS = np.copy(d)
    NMS[0, :] = NMS[W2 - 1, :] = NMS[:, 0] = NMS[:, H2 - 1] = 0
    for i in range(1, W2 - 1):
        for j in range(1, H2 - 1):

            if d[i, j] == 0:
                NMS[i, j] = 0
            else:
                gradX = dx[i, j]
                gradY = dy[i, j]
                gradTemp = d[i, j]

                # 如果Y方向幅度值较大
                if np.abs(gradY) > np.abs(gradX):
                    weight = np.abs(gradX) / np.abs(gradY)
                    grad2 = d[i - 1, j]
                    grad4 = d[i + 1, j]
                    # 如果x,y方向梯度符号相同
                    if gradX * gradY > 0:
                        grad1 = d[i - 1, j - 1]
                        grad3 = d[i + 1, j + 1]
                    # 如果x,y方向梯度符号相反
                    else:
                        grad1 = d[i - 1, j + 1]
                        grad3 = d[i + 1, j - 1]

                # 如果X方向幅度值较大
                else:
                    weight = np.abs(gradY) / np.abs(gradX)
                    grad2 = d[i, j - 1]
                    grad4 = d[i, j + 1]
                    # 如果x,y方向梯度符号相同
                    if gradX * gradY > 0:
                        grad1 = d[i + 1, j - 1]
                        grad3 = d[i - 1, j + 1]
                    # 如果x,y方向梯度符号相反
                    else:
                        grad1 = d[i - 1, j - 1]
                        grad3 = d[i + 1, j + 1]

                gradTemp1 = weight * grad1 + (1 - weight) * grad2
                gradTemp2 = weight * grad3 + (1 - weight) * grad4
                if gradTemp >= gradTemp1 and gradTemp >= gradTemp2:
                    NMS[i, j] = gradTemp
                else:
                    NMS[i, j] = 0

    # plt.imshow(NMS, cmap = "gray")

    # step4. 双阈值算法检测、连接边缘
    W3, H3 = NMS.shape
    print("NMS")
    print(W3, H3)
    DT = np.zeros([W3, H3])
    # 定义高低阈值
    TL = 0.2 * np.max(NMS)
    TH = 0.3 * np.max(NMS)
    for i in range(1, W3 - 1):
        for j in range(1, H3 - 1):
            if NMS[i, j] < TL:
                DT[i, j] = 0
            elif NMS[i, j] > TH:
                DT[i, j] = 1
                r, g, b = dstImg[i][j]
                if (r, g, b) == (255, 0, 0):
                    dstImg[i][j] = (0, 255, 0)
            elif ((NMS[i - 1, j - 1:j + 1] < TH).any() or (NMS[i + 1, j - 1:j + 1]).any()
                  or (NMS[i, [j - 1, j + 1]] < TH).any()):
                DT[i, j] = 1
                r, g, b = dstImg[i][j]
                if (r, g, b) == (255, 0, 0):
                    dstImg[i][j] = (0, 255, 0)
    print(DT.size)
    plt.imshow(DT)
    plt.savefig("Canny.png")

    for i in range(dstImg.shape[0]):
        for j in range(dstImg.shape[1]):
            # set whitematter as red
            judge = 0
            # 若该点为边缘，附近九宫格具有白质，则将该点标记为白质边缘
            if DT[i, j] == 1.0:
                for m in (-1, 2):
                    for n in (-1, 2):
                        r, g, b = dstImg[i + m, j + n]
                        if (r, g, b) == (255, 0, 0):
                            judge = 1
            if judge:
                dstImg[i][j] = (0, 255, 0)
    # plt.imshow(dstImg)
    # dstImg.save("end.png")
    plt.imshow(DT, cmap="gray")
    plt.show()
    plt.imshow(dstImg)
    plt.savefig("end.png")


if __name__ == '__main__':
    imgName = "072.png"
    # dst(imgName, threshSym4(imgName))

    # Canny
    Canny(dst(imgName, threshSym4(imgName)))
