///第三步：计算梯度值和方向 
//同样可以用不同的检测器加上把图像放到0-255之间//  
 /    P[i,j]=(S[i+1,j]-S[i,j]+S[i+1,j+1]-S[i,j+1])/2     /  
 /    Q[i,j]=(S[i,j]-S[i,j+1]+S[i+1,j]-S[i+1,j+1])/2     /  
 / 
    double *P=new double[nWidth*nHeight];
    double *Q=new double[nWidth*nHeight];
    int *M=new int[nWidth*nHeight];
    //IplImage * M;//梯度结果
    //M=cvCreateImage(cvGetSize(ColorImage),ColorImage->depth,1);
    double *Theta=new double[nWidth*nHeight];
    int nwidthstep=pCanny->widthStep;
    for(int iw=0;iw<nWidth-1;iw++)
    {
        for (int jh=0;jh<nHeight-1;jh++)
        {
            P[jh*nWidth+iw]=(double)(pCanny->imageData[min(iw+1,nWidth-1)+jh*nwidthstep]-pCanny->imageData[iw+jh*nwidthstep]+
            pCanny->imageData[min(iw+1,nWidth-1)+min(jh+1,nHeight-1)*nwidthstep]-pCanny->imageData[iw+min(jh+1,nHeight-1)*nwidthstep])/2;
            Q[jh*nWidth+iw]=(double)(pCanny->imageData[iw+jh*nwidthstep]-pCanny->imageData[iw+min(jh+1,nHeight-1)*nwidthstep]+
            pCanny->imageData[min(iw+1,nWidth-1)+jh*nwidthstep]-pCanny->imageData[min(iw+1,nWidth-1)+min(jh+1,nHeight-1)*nwidthstep])/2;
         }
     }
 //计算幅值和方向
     for(int iw=0;iw<nWidth-1;iw++)
     {
         for (int jh=0;jh<nHeight-1;jh++)
         {
            M[jh*nWidth+iw]=(int)sqrt(P[jh*nWidth+iw]*P[jh*nWidth+iw]+Q[jh*nWidth+iw]*Q[jh*nWidth+iw]+0.5);
            Theta[jh*nWidth+iw]=atan2(Q[jh*nWidth+iw],P[jh*nWidth+iw])*180/3.1415;
            if (Theta[jh*nWidth+iw]<0)
            {
                Theta[jh*nWidth+iw]=360+Theta[jh*nWidth+iw];
            }
         }
     }