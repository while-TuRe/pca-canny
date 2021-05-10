 //第四步：非极大值抑制
 //注意事项 权重的选取，也就是离得近权重大
 / 
     IplImage * N;//非极大值抑制结果
     N=cvCreateImage(cvGetSize(ColorImage),ColorImage->depth,1);
     IplImage * OpencvCannyimg;//非极大值抑制结果
     OpencvCannyimg=cvCreateImage(cvGetSize(ColorImage),ColorImage->depth,1);
     int g1=0, g2=0, g3=0, g4=0;                            //用于进行插值，得到亚像素点坐标值   
     double dTmp1=0.0, dTmp2=0.0;                           //保存两个亚像素点插值得到的灰度数据 
     double dWeight=0.0;                                    //插值的权重  
 
     for(int i=1;i<nWidth-1;i++)
     {
         for(int j=1;j<nHeight-1;j++)
         {
             //如果当前点梯度为0，该点就不是边缘点
             if (M[i+j*nWidth]==0)
             {
                 N->imageData[i+j*nwidthstep]=0;
             }else
             {
                 首先判断属于那种情况，然后根据情况插值///  
                 第一种情况///  
                 /       g1  g2                  /  
                 /           C                   /  
                 /           g3  g4              /  
                 /  
                 if((Theta[i+j*nWidth]>=90&&Theta[i+j*nWidth]<135)||(Theta[i+j*nWidth]>=270&&Theta[i+j*nWidth]<315))
                 {
                    g1=M[i-1+(j-1)*nWidth];
                    g2=M[i+(j-1)*nWidth];
                    g3=M[i+(j+1)*nWidth];
                    g4=M[i+1+(j+1)*nWidth];
                    dWeight=fabs(P[i+j*nWidth])/fabs(Q[i+j*nWidth]); 
                    dTmp1=g1*dWeight+(1-dWeight)*g2;
                    dTmp2=g4*dWeight+(1-dWeight)*g3;
                    第二种情况///  
                    /       g1                      /  
                    /       g2  C   g3              /  
                    /               g4              /  
                    /  
                 }else if((Theta[i+j*nWidth]>=135&&Theta[i+j*nWidth]<180)||(Theta[i+j*nWidth]>=315&&Theta[i+j*nWidth]<360))
                 {
                     g1=M[i-1+(j-1)*nWidth];
                     g2=M[i-1+(j)*nWidth];
                     g3=M[i+1+(j)*nWidth];
                     g4=M[i+1+(j+1)*nWidth];
                     dWeight=fabs(Q[i+j*nWidth])/fabs(P[i+j*nWidth]); 
                     dTmp1=g1*dWeight+(1-dWeight)*g2;
                     dTmp2=g4*dWeight+(1-dWeight)*g3;
                     第三种情况///  
                     /           g1  g2              /  
                     /           C                   /  
                     /       g4  g3                  /  
                     /  
                 }else if((Theta[i+j*nWidth]>=45&&Theta[i+j*nWidth]<90)||(Theta[i+j*nWidth]>=225&&Theta[i+j*nWidth]<270))
                 {
                     g1=M[i+1+(j-1)*nWidth];
                     g2=M[i+(j-1)*nWidth];
                     g3=M[i+(j+1)*nWidth];
                     g4=M[i-1+(j+1)*nWidth];
                     dWeight=fabs(P[i+j*nWidth])/fabs(Q[i+j*nWidth]); 
                     dTmp1=g1*dWeight+(1-dWeight)*g2;
                     dTmp2=g4*dWeight+(1-dWeight)*g3;
                     第四种情况///  
                     /               g1              /  
                     /       g4  C   g2              /  
                     /       g3                      /  
                     /  
                 }else if((Theta[i+j*nWidth]>=0&&Theta[i+j*nWidth]<45)||(Theta[i+j*nWidth]>=180&&Theta[i+j*nWidth]<225))
                 {
                     g1=M[i+1+(j-1)*nWidth];
                     g2=M[i+1+(j)*nWidth];
                     g3=M[i-1+(j)*nWidth];
                     g4=M[i-1+(j+1)*nWidth];
                     dWeight=fabs(Q[i+j*nWidth])/fabs(P[i+j*nWidth]); 
                     dTmp1=g1*dWeight+(1-dWeight)*g2;
                     dTmp2=g4*dWeight+(1-dWeight)*g3;
 
                 }
 
             }
 
             if ((M[i+j*nWidth]>=dTmp1)&&(M[i+j*nWidth]>=dTmp2))
             {
                   N->imageData[i+j*nwidthstep]=200;
 
             }else  N->imageData[i+j*nwidthstep]=0;
 
         }
     }
 
     
     //cvNamedWindow("Limteimg",0);  
     //cvShowImage("Limteimg",N);               //显示非抑制
     //cvWaitKey(0);  
     //cvDestroyWindow("Limteimg"); 