///第一步：灰度化 
    IplImage * ColorImage=cvLoadImage("c:\\photo.bmp",1);
    if (ColorImage==NULL)
    {
        printf("image read error");
        return 0;
    }
    cvNamedWindow("Sourceimg",0);  
    cvShowImage("Sourceimg",ColorImage);               // 
    IplImage * OpenCvGrayImage;
    OpenCvGrayImage=cvCreateImage(cvGetSize(ColorImage),ColorImage->depth,1);
    float data1,data2,data3;
    for (int i=0;i<ColorImage->height;i++)
    {
        for (int j=0;j<ColorImage->width;j++)
        {
            data1=(uchar)(ColorImage->imageData[i*ColorImage->widthStep+j*3+0]);
            data2=(uchar)(ColorImage->imageData[i*ColorImage->widthStep+j*3+1]);
            data3=(uchar)(ColorImage->imageData[i*ColorImage->widthStep+j*3+2]);
            OpenCvGrayImage->imageData[i*OpenCvGrayImage->widthStep+j]=(uchar)(0.07*data1 + 0.71*data2 + 0.21*data3);
        }
    }
    cvNamedWindow("GrayImage",0);  
    cvShowImage("GrayImage",OpenCvGrayImage);               //显示灰度图  
    cvWaitKey(0);  
    cvDestroyWindow("GrayImage"); 