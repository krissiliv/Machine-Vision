
"""

@author: PiaGober
"""
#First, the necessary packages are imported:

import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image #imagepocesing library (pil = pillow)
from os import listdir
from os import sys
import random
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn  #neural network
import numpy as np
from os import path

torch.manual_seed(43)
torch.use_deterministic_algorithms(True)

modelpath = "./model55.pth" #this name has to be changed to a non-existing one, if a new model should be created
#model55.pth was my best model, with only one mistake in the test runs, which is why I chose its name as an example
imagesdirpath = "./Train" #this is the name of the train data set (this folder is in the same folder as the code)
timagesdirpath = "./Test_AUGMENTED" #this is the name of the testset (this folder is in the same folder as the code)
epochs = 30 #number of training epochs

#The following is a protection against mistakes:
if ((path.exists(modelpath) == False) & (imagesdirpath == None)):
    print("Error: Training images not loaded!") 
    sys.exit()
if ((path.exists(modelpath) == True) & (timagesdirpath == None)):
    print("Error: Testing images not loaded!")
    sys.exit()
#if the path to the model (the model itself) does not exist and also
#no Imagesdirectory (no images for training) 
# OR if the path (the model) exists but there are no testing images 
#then the system should be exited


normalize = transforms.Normalize(mean=[0.5], std=[0.5]) 
#grayscale transform, transform to black and white pictures

#To take precautions for an error in the classification of the images:
transform=transforms.Compose([
    transforms.Resize(256), #"cut" pictures to the same size
    transforms.ToTensor(), #transform them to tensors
    normalize]) #normalize them (same brightness,...)


#TARGET: HatRiss, HatkeinenRiss
train_data_list=[] #create an (at the moment empty) list of training ata
target_list = [] #... and the target (label) list, to be able to assign lables to the pictures
train_data=[]  #this will be a list of batches (the input and the target stacked together)
files = listdir(imagesdirpath)
for i in files:
    if ".ini" in i: files.remove(i) #as due to the Augmentor I got an error message a few times, saying that there is an unknown picture .ini (which got created by the augmentor)
if path.exists(modelpath) is False: #the model should only be created if it does not yet exist:
    for idx, i in enumerate(range(len(files)-1), start=1):  #this gives me a list with all filenames
        f = random.choice(files) #and this picks a random element out of the files list -- if you only write this line, it can happen that the same element is taken multiple times. Therefore:
        files.remove(f) # f has to be removed from the files, after it was already chosen
        img = Image.open(imagesdirpath + "/" + f).convert('RGB') #the chosen image is opened and converted to a RGB 
        img = transforms.functional.to_grayscale(img) #Grayscale for RAM
        img_tensor = transform(img) #transform the image to a tensor, such that nn can work with it (Tensors are a generalization of matrices and are represented by n-dimensional arrays)
        train_data_list.append(img_tensor) #now that the image is prepared, it is added to the training list
        HatRiss = 1 if 'NG' in f else 0  #now the targets are defined: the function "Hatriss" gives 1 back, if NG (not good) is in the filename, else 0
        HatkeinenRiss = 1 if 'OK' in f else 0 #the second target: the function "HatkeinenRisss" dies exactly the opposite of "HatRiss"
        target = [HatRiss, HatkeinenRiss] #therefore this should be the target (consisting of 2 neurons)
        print(target) #eigther shows [0,1] or [1,0], the first if the waffle on the image is not "ripped"
        target_list.append(target) #now that the target (output/classification of the randomly chosen image) is defined, it is now added to the target list
        print("Image " + str(idx) + " out of " + str(len(listdir(imagesdirpath))-2) + " loaded!")
        if len(train_data_list) >= (len(listdir(imagesdirpath))-2) : #if there were at lease as many tensors added to the train data list, as exist in the above called train data folder 
            print("Appending data to a single list...") #then we stack everything together into a final list:
            train_data.append((torch.stack(train_data_list), target_list)) #stacks the trainingsdatalist and the list with the targets into one list (Training data image, target) <- this is a batch containing the training data and their final classification
            train_data_list = [] #empties the list, which contains the tensors of the training images, to save RAM
            print("Appending finished!")
 
            
#Now the neural net can be defined:
class Netz(nn.Module):
    def __init__(self): #initializer method, initializes/creates memory space, to remember details about the new elements, which are later defined in the class
        #self is now taken as the first argument per default
        super(Netz, self).__init__() #now I am calling the newly defined class __init__
        self.conv1 = nn.Conv2d(1, 2, kernel_size=5) #this will be the first variable, when defining an element of the __init__(self) class
        self.conv2 = nn.Conv2d(2, 4, kernel_size=5) #as I am working with pictures, I put three convolutional layers into the initializer
        self.conv3 = nn.Conv2d(4, 8, kernel_size=5) #The numbers? = dimensions of the matrices, which represent the filter
        self.fc1 = nn.Linear(7616, 1000) #takes the convolution output and returns a classification decision
        # = fully connected layer should take as an inputexactly the number of neurons, that np.shape(input) => . mal . mal .  returns (see comment below from line 114 on)
        self.fc2 = nn.Linear(1000, 2)      

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x) #activation function, relu f(x) = max(0,x),where x is the input, makes sure that the model is not too linear (which would probably lead to more errors for complex models)
        x = F.max_pool2d(x,2) #maxpooling separates the picture (which comes out of the convolutional layer rauskommt) 
        #into 2x2 pixelblocks and takes as output the maximum of each block (=> the image gets smaller)
        x = self.conv2(x) #this is done three times:
        x = F.relu(x)
        x = F.max_pool2d(x,2)
        x = self.conv3(x)        
        x = F.relu(x)
        x = F.max_pool2d(x,2)    
        x = x.view(-1, 7616) # view function reshapes tensor
        x = F.relu(self.fc1(x))
        x = F.dropout(x) #this takes a random part of the activation and destroys it. 
        #this forces the nn to train a more robust recognition and concentrate on many properties of the input, instead of just one (which could be another one as the one that we want (broken - not broken))
        x = self.fc2(x)
        return F.softmax(x, dim=1) #softmax makes the final classification-decision
        #from the fully connected layer and transforms them into probabilities (with the help of an Eulerfunktion)
        exit()

''' 
the number 7616 was found in that way:       
#____________________________________
f = random.choice(files)
img = Image.open(imagesdirpath + "/" + f).convert('RGB')
img = transforms.functional.to_grayscale(img)
img_tensor = transform(img)
conv1 = nn.Conv2d(1, 2, kernel_size=5) 
conv2 = nn.Conv2d(2, 4, kernel_size=5) 
conv3 = nn.Conv2d(4, 8, kernel_size=5)
conv1(img_tensor)
print(np.shape(F.max_pool2d(F.relu(conv3(F.max_pool2d(F.relu(conv2(F.max_pool2d(F.relu(conv1(img_tensor)),2))),2))),2)))
#____________________________________
'''  
  
        
#now I generate the model as a "Netz":
model = Netz()


optimizer = optim.RMSprop(model.parameters(), lr = 0.001) #fix the learning rate

def train(epoch): #now it will be defined, what should happen during a training epoch:
    model.train() #model is set to train, such that it is trained and not just executed
    for idx, (data, target) in enumerate(train_data): #if the index of the chosen picture is in the enumeration of the train_data.append batchlist - defined in line 77 - then:
        target = torch.Tensor(target) #it is again transformed to a tensor, such that the nn can work with it
        data = Variable(data) #1.Batch entry: data (so the image) is taken as a variable
        target = Variable(target) #2.Batch entry: target. --> data and target have to be taken as variables
        optimizer.zero_grad() #set radient to 0, before the backpropagation starts, because otherwise the gradient does not point into the direction of the extremum
        out = model(data) #output should be the output of the defined model (in class "Netz"), if I "feed" it with data
        criterion = F.binary_cross_entropy #the criterion is, that the images should be subdivided into two classes
        loss = criterion(out, target) #Loss should be calculated as the criterion of output(current/guessed result) und target (predetermined actual result)
        loss.backward() #Loss should be back - propagated (=> the weights of the neuron-connections are changed depending on their impact on the error)  
        optimizer.step() #now the optimizer has to be told to take a step
        


epoch_num = range(0,int(epochs))

if path.exists(modelpath) == False: #check, if the model is already trained, so if it exists
    for idx, epoch in enumerate(epoch_num, start=1): #I want it to be trained for 30 epochs, if it does not exist yet
        print("Epoch " + str(idx) + " out of " + str(max(epoch_num)+1) + " started!")
        train(epoch) #and then the function train is used to train the model
        print("Epoch " + str(idx) + " out of " + str(max(epoch_num)+1) + " finished!")
    torch.save(model, modelpath) #when it is done training, the model is saved
else: #else, if the model exists:
    model = torch.load(modelpath) #it is loaded
    model.eval() #.. evaluated
    vals = [] #an (currently empty) value set is created
    tests = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] #...the testsetindices (the testset contains images; enumerated from 1-10)
    result = [] #and an (currently empty) result set is created
    if timagesdirpath is None: sys.exit() #if there is no testimageset (if it does not exist), the system should be exited
    else: #else it should be checked, if the model is goog and if it can be used for correct classifications
        for i in listdir(timagesdirpath): #so for the images in the testset:
            if ".ini" in i : continue #as above: we need to remove .ini, if it exists (it was probably "unwillingly" created by the augmentor)
            test_img = Image.open(timagesdirpath + "/" + i) #for the other, correct formats (images), (every one except for .ini) the images are opened
            test_img = transforms.functional.to_grayscale(test_img) #... then transformed to grayscale due to RAM
            test_img_tensor = transform(test_img) #with the pre-defined transform-funktion they are transformed into the correct format (amongst others to a tensor)
            loading_image = model(test_img_tensor[None, ...]) #... afterwards the tensor is sent to our model 
            cpu_pred = loading_image.cpu() #and along with this to the cpu-prozessor
            numpyresult = cpu_pred.data.numpy() #there, the image that ran through the model, is transformed to an array
            listresult = np.ndarray.tolist(numpyresult[0]) #and the n-dim array (numpyresult) is transformed to a list
            endresult = str(listresult).split(", ") 
            bad = endresult[0].replace("[", "") #if the waffle is "bad"/broken, is determined by the first value (which has index 0 in the endresult), see HatRiss (above)
            good = endresult[1].replace("]", "") #if the waffle is "good" ist, is determined by the second value (which has index 1 in the endresult), see HatkeinenRiss (above)
            if good > bad: vals.append(1) #if the model thinks that the waffle is good, then add 1 to the output list (=List "vals")
            else: vals.append(0) # else (if the model thinks, that the waffle is broken) add 0
            #(the list vals will later be used to check how good the performance of our model is)
        for l, f in zip(vals,listdir(timagesdirpath)): #if l is in the list "vals", and f is a testimage, then:
            if ".ini" in f: continue #as always: if there is .ini in f, ignore f, else:
            if int(f[:2].replace(".", "")) in [1, 2, 3, 4, 5]: #delete the . from the name of f and check, if it is a waffle with index 1-5 
                #The testimages 1-5 in the testset are waffles, which are broken
                if l == 0: result.append(1) #so if the value for l from "vals" is in fact 0, then this means that 
                #good<bad, so the waffle is probably bad (acc to the model) and therefore the image was classifyed correctly
                else: result.append(0) #else it was wrongly predicted
            if int(f[:2].replace(".", "")) in [6, 7, 8, 9, 10]: #the images 6-10 are good waffles, so:
                if l == 1: result.append(1) #if the value l is predicted to be 1 (good>bad), then the model was right, therefore the output should be 1
                else: result.append(0) #else it was wrong and 0 should be the output
        plt.scatter(tests, result,  color='black') #now a scatter is being built, which sets test set in relation to the results
        plt.plot(tests, result, color='blue', linewidth=3) #.. with this a plot is created with horizontal axis=tests und vertical axis=result set
        plt.show() #... and the plot is being compiled to show, how good the model was
   
