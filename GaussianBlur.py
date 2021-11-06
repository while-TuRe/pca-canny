///第二步：高斯滤波 
///
    double nSigma=0.2;
    int nWindowSize=1+2*ceil(3*nSigma);//通过sigma得到窗口大小
    int nCenter=nWindowSize/2;
    int nWidth=OpenCvGrayImage->width;
    int nHeight=OpenCvGrayImage->height;
    IplImage * pCanny;
    pCanny=cvCreateImage(cvGetSize(ColorImage),ColorImage->depth,1);
    //生成二维滤波核
    double *pKernel_2=new double[nWindowSize*nWindowSize];
    double d_sum=0.0;
    for(int i=0;i<nWindowSize;i++)
    {
        for (int j=0;j<nWindowSize;j++)
        {
            double n_Disx=i-nCenter;//水平方向距离中心像素距离
            double n_Disy=j-nCenter;
            pKernel_2[j*nWindowSize+i]=exp(-0.5*(n_Disx*n_Disx+n_Disy*n_Disy)/(nSigma*nSigma))/(2.0*3.1415926)*nSigma*nSigma; 
            d_sum=d_sum+pKernel_2[j*nWindowSize+i];
        }
    }
    for(int i=0;i<nWindowSize;i++)//归一化处理
    {
        for (int j=0;j<nWindowSize;j++)
        {
          pKernel_2[j*nWindowSize+i]=pKernel_2[j*nWindowSize+i]/d_sum;
        }
    }
    //输出模板
    for (int i=0;i<nWindowSize*nWindowSize;i++)
    {
        if (i%(nWindowSize)==0)
        {
          printf("\n");
        }
        printf("%.10f ",pKernel_2[i]);
    }
    //滤波处理
    for (int s=0;s<nWidth;s++)
    {
        for (int t=0;t<nHeight;t++)
        {
            double dFilter=0;
            double dSum=0;
            //当前坐标（s,t）
            //获取8邻域
            for (int x=-nCenter;x<=nCenter;x++)
            {
                for (int y=-nCenter;y<=nCenter;y++)
                {
                    if ((x+s>=0)&&(x+s<nWidth)&&(y+t>=0)&&(y+t<nHeight))//判断是否越界
                    {
                        double currentvalue=(double)OpenCvGrayImage->imageData[(y+t)*OpenCvGrayImage->widthStep+x+s];
                        dFilter+=currentvalue*pKernel_2[x+nCenter+(y+nCenter)*nCenter];
                        dSum+=pKernel_2[x+nCenter+(y+nCenter)*nCenter];
                    }
                }
            }
            pCanny->imageData[t*pCanny->widthStep+s]=(uchar)(dFilter/dSum);
        }
    }
   


    cvNamedWindow("GaussImage",0);  
    cvShowImage("GaussImage",pCanny);               //显示高斯图
    cvWaitKey(0);  
    cvDestroyWindow("GaussImage"); 