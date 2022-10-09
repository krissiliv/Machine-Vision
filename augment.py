from imgaug import augmenters as iaa
import cv2
from os import listdir
import random
#import argparse
from os import sys
from os import path

#DOCUMENTATION: https://imgaug.readthedocs.io/en/latest/
# https://github.com/aleju/imgaug

#parser = argparse.ArgumentParser()
#parser.add_argument('--imagesdir', help='imagesdir IMAGESDIR', required=True)
#parser.add_argument('--outputdir', help='outputdir OUTPUTDIR', required=True)
#parser.add_argument('--imgnum', help='imgnum IMGNUM', required=True, type=int)
#args = parser.parse_args()

imagesdir = "./Train"
outputdir = "./augoutput"
imagenumber = 10

if ((path.exists(imagesdir)) & (len(listdir(imagesdir)) > 0)): files = listdir(imagesdir)
else: 
    print("Error: Directory empty or invalid!")
    sys.exit()
 
for i in files:
    if i[-3:] != "bmp": files.remove(i)    
 
images = []
namelist = []
#DATA LOADING
for idx, i in enumerate(range(len(files))):
    f = files[idx]
    img = cv2.imread(imagesdir + "/" + str(f))
    namelist.append(str(f))
    images.append(img)
    print("File " + str(idx + 1) + " out of " + str(len(listdir(imagesdir))-1) + " loaded!")
print(namelist)
#AUGMENTATION PARAMETERS
seq = iaa.Sequential(([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.PerspectiveTransform(scale=(0.03, 0.03))
    ]))

imgnum = len(listdir(outputdir)) + imagenumber
print("Image augmentation started!")
while len(listdir(outputdir)) < imgnum:
    images_aug = seq.augment_images(images)
    for o,a in zip(images_aug, namelist):
        if len(listdir(outputdir)) < imgnum:
            num = random.randint(1, 999999)
            while (outputdir + '/aug_' + str(num) + '.bmp') in listdir(outputdir): num = random.randint(1, 999999)
            cv2.imwrite(outputdir + '/aug_' + str(num) + '_' + a + '.bmp', o)
        else: break
print("Image augmentation finished!")
