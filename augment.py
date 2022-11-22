from imgaug import augmenters as iaa
import cv2
from os import listdir
import random
#import argparse
from os import sys
from os import path

#DOCUMENTATION: https://imgaug.readthedocs.io/en/latest/
# https://github.com/aleju/imgaug

imagesdir = "./Train" #the training images for the NN
outputdir = "./augoutput" #a folder created for the output of the augmentor
imagenumber = 10 #the number of images that should be created

if ((path.exists(imagesdir)) & (len(listdir(imagesdir)) > 0)): files = listdir(imagesdir) #if the path to the trainin images exists and if it actually contains images, 
#then we define "files" as the directory of the training images
else: 
    print("Error: Directory empty or invalid!") #else, this error messae should inform us to check the above
    sys.exit() #and the system should be exited
 
for i in files:
    if i[-3:] != "bmp": files.remove(i) #if the file ends with the letters "bmp" then it should be removed from the list (it is an image that should not be used for data augmentation, as it is (probably) an already augmented image) 
 
images = [] # we create an image list and a
namelist = [] #name list
#DATA LOADING
for idx, i in enumerate(range(len(files))): #and for all the files, which should be used for data augmentation
    f = files[idx] #we pick one and call it f
    img = cv2.imread(imagesdir + "/" + str(f)) #cv2 is a function that ca read images, insert the chosen image
    namelist.append(str(f)) #and append the namelist with the name of the chosen image's name 
    images.append(img) #and the imageslist with the actual image
    print("File " + str(idx + 1) + " out of " + str(len(listdir(imagesdir))-1) + " loaded!")
print(namelist) #after we did this for every file, we print the namelist of the images
#AUGMENTATION PARAMETERS
seq = iaa.Sequential(([
        iaa.Fliplr(0.5), #we let the imageaugmentor flip each image with a chance of 50% to the left or right
        iaa.Flipud(0.5), #and the same for up or down
        iaa.PerspectiveTransform(scale=(0.03, 0.03)) #changes the perspective of the image, using the distance of each image point from the corners
    ]))

imgnum = len(listdir(outputdir)) + imagenumber #check how many images are already in the augoutput-folder and add "imagenumber" more images. This number is now fixed for the rest of the code
#AUGMETATION PROCESSING AND SAVING
print("Image augmentation started!")
while len(listdir(outputdir)) < imgnum: #as long as there are only less than "imagenumber" images added
    images_aug = seq.augment_images(images) #let the training images (from the Train folder) run through the augmentor
    for o,a in zip(images_aug, namelist): #and for all the outputs
        if len(listdir(outputdir)) < imgnum:
            num = random.randint(1, 999999) #choose a random number (it is used below to create a name for the new image)
            while (outputdir + '/aug_' + str(num) + '.bmp') in listdir(outputdir): num = random.randint(1, 999999) #check if random number was used and if so, take another one
            cv2.imwrite(outputdir + '/aug_' + str(num) + '_' + a + '.bmp', o) #start augmenting and save augmented image (it will - amongst others - include the name "a" of the image that was used for augmentation, 
            #for example to make sure it has the "NG" or "OK" in the filename, so that it can later be used for training the neural network - if needed)
        else: break
print("Image augmentation finished!")
