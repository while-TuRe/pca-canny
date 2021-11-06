///第五步：双阈值的选取
 //注意事项  注意是梯度幅值的阈值  
 / 
   int nHist[1024];//直方图
   int nEdgeNum;//所有边缘点的数目
   int nMaxMag=0;//最大梯度的幅值
   for(int k=0;k<1024;k++)
   {
       nHist[k]=0;
   }
   for (int wx=0;wx<nWidth;wx++)
   {
       for (int hy=0;hy<nHeight;hy++)
       {
           if((uchar)N->imageData[wx+hy*N->widthStep]==200)
           {
               int Mindex=M[wx+hy*nWidth];
               nHist[M[wx+hy*nWidth]]++;//获取了梯度直方图
               
           }
       }
   }
   nEdgeNum=0;
   for (int index=1;index<1024;index++)
   {
       if (nHist[index]!=0)
       {
           nMaxMag=index;
       }
       nEdgeNum+=nHist[index];//经过non-maximum suppression后有多少边缘点像素  
   }
 //计算两个阈值 注意是梯度的阈值
   int nThrHigh;
   int nThrLow;
   double dRateHigh=0.7;
   double dRateLow=0.5;
   int nHightcount=(int)(dRateHigh*nEdgeNum+0.5);
   int count=1;
   nEdgeNum=nHist[1];
   while((nEdgeNum<=nHightcount)&&(count<nMaxMag-1))
   {
       count++;
       nEdgeNum+=nHist[count];
   }
   nThrHigh=count;
   nThrLow= (int)(nThrHigh*dRateLow+0.5);
   printf("\n直方图的长度 %d \n  ",nMaxMag);
   printf("\n梯度的阈值幅值大 %d 小阈值 %d \n  ",nThrHigh,nThrLow);