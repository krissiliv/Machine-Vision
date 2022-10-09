import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image #Bildverarbeitungs library (pil = pillow)
from os import listdir
from os import sys
import random
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
#import argparse
from os import path

torch.manual_seed(43)
torch.use_deterministic_algorithms(True)
    
#parser = argparse.ArgumentParser()
#parser.add_argument('--model', help='model MODEL', required=True)
#parser.add_argument('--imagesdir', help='imagesdir IMAGESDIR')
#parser.add_argument('--epochs', help='epochs EPOCHS', default=30)
#parser.add_argument('--timagesdir', help='timagesdir TIMAGESDIR')
#args = parser.parse_args()

modelpath = "./model55.pth"
imagesdirpath = "./Train" #Trainingsset
timagesdirpath = "./Test_AUGMENTED" #Testset
epochs = 30

if ((path.exists(modelpath) == False) & (imagesdirpath == None)):
    print("Error: Training images not loaded!") 
    sys.exit()
if ((path.exists(modelpath) == True) & (timagesdirpath == None)):
    print("Error: Testing images not loaded!")
    sys.exit()

normalize = transforms.Normalize(mean=[0.5], std=[0.5]) 

transform=transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    normalize])

train_data_list=[]
target_list = []
train_data=[]
files = listdir(imagesdirpath) 
for i in files:
    if ".ini" in i: files.remove(i)
if path.exists(modelpath) is False:
    for idx, i in enumerate(range(len(files)-1), start=1): 
        f = random.choice(files)
        files.remove(f)
        img = Image.open(imagesdirpath + "/" + f).convert('RGB')
        img = transforms.functional.to_grayscale(img)
        img_tensor = transform(img)
        train_data_list.append(img_tensor)
        HatRiss = 1 if 'NG' in f else 0
        HatkeinenRiss = 1 if 'OK' in f else 0
        target = [HatRiss, HatkeinenRiss]
        print(target)
        target_list.append(target)
        print("Image " + str(idx) + " out of " + str(len(listdir(imagesdirpath))-2) + " loaded!")
        if len(train_data_list) >= (len(listdir(imagesdirpath))-2) :  
            print("Appending data to a single list...") 
            train_data.append((torch.stack(train_data_list), target_list)) 
            train_data_list = [] 
            print("Appending finished!")
            
class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=5)
        self.conv2 = nn.Conv2d(2, 4, kernel_size=5) 
        self.conv3 = nn.Conv2d(4, 8, kernel_size=5)
        self.fc1 = nn.Linear(7616, 1000)
        self.fc2 = nn.Linear(1000, 2)
        
    def forward(self, x): ##
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2) 
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2)    
        x = x.view(-1, 7616)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)
        exit()

model = Netz()

optimizer = optim.RMSprop(model.parameters(), lr = 0.001)

def train(epoch):
    model.train()
    for idx, (data, target) in enumerate(train_data):
        target = torch.Tensor(target) 
        data = Variable(data) 
        target = Variable(target)
        optimizer.zero_grad()
        out = model(data)
        criterion = F.binary_cross_entropy
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

epoch_num = range(0,int(epochs))

if path.exists(modelpath) == False:
    for idx, epoch in enumerate(epoch_num, start=1):
        print("Epoch " + str(idx) + " out of " + str(max(epoch_num)+1) + " started!")
        train(epoch)
        print("Epoch " + str(idx) + " out of " + str(max(epoch_num)+1) + " finished!")
    torch.save(model, modelpath)
else:
    model = torch.load(modelpath)
    model.eval()
    vals = []
    tests = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = []
    if timagesdirpath is None: sys.exit()
    else: 
        for i in listdir(timagesdirpath):
            if ".ini" in i : continue
            test_img = Image.open(timagesdirpath + "/" + i) 
            test_img = transforms.functional.to_grayscale(test_img) 
            test_img_tensor = transform(test_img)
            loading_image = model(test_img_tensor[None, ...]) 
            cpu_pred = loading_image.cpu()
            numpyresult = cpu_pred.data.numpy()
            listresult = np.ndarray.tolist(numpyresult[0])
            endresult = str(listresult).split(", ")
            bad = endresult[0].replace("[", "")
            good = endresult[1].replace("]", "")
            if good > bad: vals.append(1) 
            else: vals.append(0)
          
        for l, f in zip(vals,listdir(timagesdirpath)):
            if ".ini" in f: continue 
            if int(f[:2].replace(".", "")) in [1, 2, 3, 4, 5]:
                if l == 0: result.append(1)
                else: result.append(0)
            if int(f[:2].replace(".", "")) in [6, 7, 8, 9, 10]:
                if l == 1: result.append(1)
                else: result.append(0)
        plt.scatter(tests, result,  color='black')
        plt.plot(tests, result, color='blue', linewidth=3)
        plt.show()
        
