from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

########## open the image ##########
im = Image.open('C:\\Users\\25921\\Desktop\\codefield\\MRI_greywhite\\no-eyeball-ps.png')
rgb_im = im.convert('RGB')#convert RGBA to RGB
rgb_im.show() #it turns out the pics aren't segmented well,they just cover the skull and the eye balls
########## open the image ##########

########## draw histogram ##########
r,g,b = rgb_im.split() #split R/G/B
plt.figure("flowerHist")
ar = np.array(r).flatten()  #flatten numpy into 1D_array
plt.hist(ar,bins = 256, density=False,facecolor="green", edgecolor="green",alpha=0.5)
########## draw histogram ##########

########## manually count the number of pixels of every grayscale from 0 to 255 ##########
xArray=range(0,256) #grayscale from 0 to 255
countArray = [0 for i in range (0,256)] #countArray to record the number of pixels in different grayscale
pixelNum = len(ar)
for i in range(0,256):
    for j in range(0,pixelNum):
        if(ar[j] == i):
            countArray[i] += 1
countArray[0]=0 #set the number of pixel_0 as 0 to remove its effect on the outcome
########## manually count the number of pixels of every grayscale from 0 to 255 ##########

########## 15-polynomial fitting  ##########
fit = np.polyfit(xArray, countArray, 15) 
factor = np.poly1d(fit) 
yPoly=factor(xArray) #polynomial
########## 15-polynomial fitting  ##########

########## to draw the histogram and the fitted polynomial  ##########
plt.plot(xArray, yPoly, 'r',label='polyfit values',color='red')
plt.show()
########## to draw the histogram and the fitted polynomial  ##########

########## to find the appropriate threshold to seg  ##########
peaks_lower = signal.argrelextrema(yPoly,np.less)[0] # minimum numpy
peaks=signal.find_peaks(yPoly, distance=10) # maximum numpy
peakList_Index=peaks[0].tolist() #turn maximum numpy into list

print("The minimum list of the grayscale is as below")
print(peaks_lower)
print("The maximum list of the grayscale is as below")
print(peakList_Index)

peakList=[0 for i in range(0,len(peakList_Index))]
for i in range(0,len(peakList_Index)):
    peakList[i]=yPoly[peakList_Index[i]]

####figure out the top 2 grayscale
peak_1=max(peakList)
peakIndex_1=peakList.index(peak_1)

peakList[peakIndex_1]=0

peak_2=max(peakList)
peakList[peakIndex_1]=peak_1
peakIndex_2=peakList.index(peak_2)

lowerLimit=peakList_Index[peakIndex_2]
higherlimit=peakList_Index[peakIndex_1]
####figure out the top 2 grayscale

threshold_tmp=0
threshold=peaks_lower[0]
for i in range(0,len(peaks_lower)):
    if peaks_lower[i]>lowerLimit and peaks_lower[i]<higherlimit:
        threshold_tmp=peaks_lower[i]
        if threshold<lowerLimit or threshold>higherlimit or yPoly[threshold] >= yPoly[threshold_tmp]:
            threshold = threshold_tmp

print("the threshold turns out to be:")
print(threshold)
########## to find the appropriate threshold to seg  ##########

########## use the threshold to seg  ##########
for i in range(rgb_im.size[0]):
    for j in range(rgb_im.size[1]):
        r, g, b = rgb_im.getpixel((i, j))
        if r >= threshold:	
            rgb_im.putpixel((i,j),(0,255,0))#set whitematter as green
########## use the threshold to seg  ##########

########## show the outcome  ##########
rgb_im.show()
########## show the outcome  ##########
