<<<<<<< HEAD

import os
from urllib.request import urlretrieve
#sides = ['axl','sag','cor']
sides = ['cor']
def download():
    for side in sides:
        if not os.path.exists('./pictures/'+side):
            os.mkdir('./pictures/'+side)
        for i in range(0,127):#[0..126]
            IMAGE_URL = 'http://www.med.harvard.edu/AANLIB/cases/caseNA/gr/'+side+'/'+str(i).zfill(3)+'.png'
            print(IMAGE_URL)
            urlretrieve(IMAGE_URL, './pictures/'+side+'/'+str(i).zfill(3)+'.png')     
if __name__=='__main__':
=======

import os
from urllib.request import urlretrieve
#sides = ['axl','sag','cor']
sides = ['cor']
def download():
    for side in sides:
        if not os.path.exists('./pictures/'+side):
            os.mkdir('./pictures/'+side)
        for i in range(0,127):#[0..126]
            IMAGE_URL = 'http://www.med.harvard.edu/AANLIB/cases/caseNA/gr/'+side+'/'+str(i).zfill(3)+'.png'
            print(IMAGE_URL)
            urlretrieve(IMAGE_URL, './pictures/'+side+'/'+str(i).zfill(3)+'.png')     
if __name__=='__main__':
>>>>>>> f4dd958ae4d6d16accabef660fabaddd0c1ec2c4
    download()