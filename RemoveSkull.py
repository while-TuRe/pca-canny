import numpy as np
import cv2 as cv
import os
from urllib.request import urlretrieve
from PIL import Image
MF = np.array(((0, 1, 1, 1,0), (1, 1, 1, 1, 1), (1, 1, 1, 1, 1), (1, 1, 1, 1, 1), (0, 1, 1, 1, 0)), dtype=int)

def otsu(img_gray):
    """
    img_gray: cv类型的灰度图
    return：cv类型的二值图
    """
    ret, img_bin = cv.threshold(img_gray, 0, 255, cv.THRESH_OTSU)
    return ret,img_bin
    
def fillHole(img_bin, kernel=np.ones((3, 3))):
    cur_marker = None
    pre_marker = np.zeros(img_bin.shape)
    height, width = img_bin.shape
    pre_marker[[0, height - 1]] = 1 - img_bin[[0, height - 1]]
    pre_marker[:, [0, width - 1]] = 1 - img_bin[:, [0, width - 1]]

    pre_marker = arrayToCV(pre_marker)
    mask = 1 - img_bin
    while True:
        cur_marker = cv.bitwise_and(cv.dilate(src=pre_marker, kernel=kernel), mask)
        difference = cv.subtract(cur_marker, pre_marker)
        if not np.any(difference):
            break
        pre_marker = cur_marker
    return 1 - cur_marker
def regionGrowingToGetBrain(img_cv, img_bin, seed, threshold: int, greg_threshold: int):
    """
    用区域生长算法取出颅骨
    img_cv: cv类型
    seed: 区域生长的种子点  二元元组 （row, col）
    threshold: 像素点的差小于该阈值表明两个点属于同一类
    return: cv
    """
    # cv类型转为array
    img_array = CVToarray(img_bin)
    img_array_src = CVToarray(img_cv)
    # 转为int，否则后面相减会溢出
    img_array = np.array(img_array, dtype=int)
    row, col = img_array.shape
    # 区域生长后得到的新图像  二值图  初始化为0
    filtered_region = np.zeros((row, col), dtype=int)
    filtered_region[seed] = 255
    container = [seed]
    container_grey = []
    def deal(new_seed, ext_seed):
        if filtered_region[ext_seed] == 0:
            if abs(img_array[ext_seed] - img_array[new_seed]) <= threshold:
                filtered_region[ext_seed] = 255
                container.append(ext_seed)
            elif img_array_src[ext_seed] < greg_threshold and img_array_src[ext_seed]>0:
                filtered_region[ext_seed] = 255
                container_grey.append(ext_seed)
                 
    # 广度优先进行生长
    while container:
        # 新的生长点
        new_seed = container.pop(0)
        if new_seed[0] - 1 >= 0:
            # 上
            ext_seed = (new_seed[0] - 1, new_seed[1])
            deal(new_seed, ext_seed)
        if new_seed[0] + 1 < row:
            # 下
            ext_seed = (new_seed[0] + 1, new_seed[1])
            deal(new_seed, ext_seed)
        if new_seed[1] - 1 >= 0:
            # 左
            ext_seed = (new_seed[0], new_seed[1] - 1)
            deal(new_seed, ext_seed)
        if new_seed[1] + 1 < col:
            # 右
            ext_seed = (new_seed[0], new_seed[1] + 1)
            deal(new_seed, ext_seed)
    # 转为cv类型后返回
    return arrayToCV(filtered_region),container_grey
    
def BrainGreyStretch(img_cv, filtered_region,container_grey):
    """
    用区域生长算法进行延伸
    img_cv: cv类型灰度图
    filtered_region：待延伸二值图
    container_grey：区域生长算法取到的边缘点
    return: cv
    """
    # cv类型转为array
    img_array_src = CVToarray(img_cv)
    row, col = img_array_src.shape
    #对于每个新点的检查方式：如果目前不在集合且灰度值递减
    def deal_grey(new_seed, ext_seed):
        if img_array_src[new_seed] > img_array_src[ext_seed] and filtered_region[ext_seed] == 0:
            filtered_region[ext_seed] = 255
            container_grey.append(ext_seed)

    #container_grey
    while container_grey:
        # 新的生长点
        new_seed = container_grey.pop(0)
        if new_seed[0] - 1 >= 0:
            # 上
            ext_seed = (new_seed[0] - 1, new_seed[1])
            deal_grey(new_seed, ext_seed)
        if new_seed[0] + 1 < row:
            # 下
            ext_seed = (new_seed[0] + 1, new_seed[1])
            deal_grey(new_seed, ext_seed)
        if new_seed[1] - 1 >= 0:
            # 左
            ext_seed = (new_seed[0], new_seed[1] - 1)
            deal_grey(new_seed, ext_seed)
        if new_seed[1] + 1 < col:
            # 右
            ext_seed = (new_seed[0], new_seed[1] + 1)
            deal_grey(new_seed, ext_seed)

    # 转为cv类型后返回
    return arrayToCV(filtered_region)

def printImg(img):
    row,col=img.shape
    for i in range(row):
        for j in range(col):
            print(img[i][j],end=" ")
        print("")

def arrayToCV(img_array):
    """
    array转cv类型
    """
    # array转PIL.Image
    image = Image.fromarray(img_array)
    # PIL.Image转换成OpenCV格式
    img_cv = cv.cvtColor(np.asarray(image, dtype='uint8'), cv.COLOR_RGB2BGR)
    return cv.cvtColor(img_cv, cv.COLOR_RGB2GRAY)

def CVToarray(img_cv):
    """
    cv转array类型
    """
    img = np.array(img_cv)
    return img

def show(cv_img):
    """
    传入cv类型，转为PIL类型显示
    """
    # OpenCV转换成PIL.Image格式
    image = Image.fromarray(cv.cvtColor(cv_img, cv.COLOR_BGR2RGB))
    image.show()

def RemoveSkull(src_path):
    # 有眼球
    src = cv.imread(src_path, 0)  # 256*256
    #show(src)
    # 二值化  自适应阈值的二值化去除了大脑特征，不建议使用
    # src_bin = adaptiveThreshold(src)
    threshold, src_bin = otsu(src)

    # 开操作，去除粘连
    src_bin = cv.morphologyEx(src_bin,cv.MORPH_OPEN,arrayToCV(MF))
    #show(src_bin)
    # 区域生长获得大脑的范围  这里的种子点是经验选取的
    brain_bin,container_grey = regionGrowingToGetBrain(src, src_bin, (100,100), 1, threshold)
    #显示边缘
    # edge = np.zeros(src.shape,int)
    # for i in container_grey:
    #     edge[i]=255
    # show(arrayToCV(edge))

    #show(brain_bin)

    #闭操作去除缝隙孔洞等，直到收敛
    while(True):
        ex_brain = brain_bin
        #brain_bin = closing(brain_bin)
        brain_bin = cv.morphologyEx(brain_bin,cv.MORPH_CLOSE,arrayToCV(MF))
        if (ex_brain == brain_bin).all():
            break
    brain_bin = fillHole(brain_bin)*255
    #show(brain_bin)
    brain_bin = BrainGreyStretch(src,brain_bin,container_grey)
    #show(brain_bin)

    #闭操作去除缝隙孔洞等，直到收敛
    while(True):
        ex_brain = brain_bin
        #brain_bin = closing(brain_bin)
        brain_bin = cv.morphologyEx(brain_bin,cv.MORPH_CLOSE,arrayToCV(MF))
        if (ex_brain == brain_bin).all():
            break

    #show(brain_bin)
    brain_bin=cv.medianBlur(brain_bin,5)

    #show(brain_bin)
    brain_bin[brain_bin == 255] = 1
    brain_bin = brain_bin*src
    #show(brain_bin*src)
    return brain_bin

sides = ['axl','sag','cor']
rangee = [(52,115), (25,75), (14,100)]
if __name__ == '__main__':
    #show(RemoveSkull('pictures/axl/072.png'))
    for k in range(len(sides)):
        for i in range(rangee[k][0],rangee[k][1]):#[0..126]
            name = str(i).zfill(3)+'.png'
            src_path = './pictures/'+sides[k]+'/'+ name
            rsl = RemoveSkull(src_path)
            src_path = './pictures/noSkull/'+sides[k]+'/' + name
            print(src_path)
            cv.imwrite(src_path,rsl)

