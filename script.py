# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 19:35:44 2020

@author: e14
"""
    
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
import torch.nn as nn  #neural network
import numpy as np
#import argparse #Eingangsargumente
from os import path

torch.manual_seed(43)
torch.use_deterministic_algorithms(True)

#Das sind die Eingangsargumente, die man auswählen muss
# falls man die Ausgabe nicht in der Console anzeigen lassen will,
# sondern im anacondaprompt-Fenster:
    
#parser = argparse.ArgumentParser()
#parser.add_argument('--model', help='model MODEL', required=True)
#parser.add_argument('--imagesdir', help='imagesdir IMAGESDIR')
#parser.add_argument('--epochs', help='epochs EPOCHS', default=30)
#parser.add_argument('--timagesdir', help='timagesdir TIMAGESDIR')
#args = parser.parse_args()

modelpath = "./model55.pth" #der Name muss auf einen nicht existierenden Namen geändert werden, falls man ein neues Modell kreieren will
#model55.pth war das beste Modell, mit nur einem Fehler in den Test runs, darum habe ich dieses als Beispiel gewählt
imagesdirpath = "./Train" #so heißt das Trainingsset
timagesdirpath = "./Test_AUGMENTED" #so heißt das Testset
epochs = 30 #Anzahl der Trainingseinheiten

#Das ist eine Absicherung für Fehler:
if ((path.exists(modelpath) == False) & (imagesdirpath == None)):
    print("Error: Training images not loaded!") 
    sys.exit()
if ((path.exists(modelpath) == True) & (timagesdirpath == None)):
    print("Error: Testing images not loaded!")
    sys.exit()
#Falls der Pfad zum Modell (d.h. das Modell selbst) nicht existiert und auch
#kein Imagesdirectory (also keine Bilder zum Trainieren) 
# ODER falls der Pfad (also das Modell) existiert aber keine Bilder zum Testen da sind
#soll das System verlassen werden


normalize = transforms.Normalize(mean=[0.5], std=[0.5]) 
#grayscale transform, also Transformierung auf S-W Bilder

#Damit Fehler in der Unterscheidung zwischen den zwei Targets vorgebeut werden:
transform=transforms.Compose([
    transforms.Resize(256), #Bilder auf dieselbe Größe zuschneiden
    transforms.ToTensor(), #Bilder als Tensoren speichern
    normalize]) #selbe Helligkeit,...

#TARGET: HatRiss, HatkeinenRiss

train_data_list=[] #erstellen die (noch leere) Liste mit den Trainingsdaten
target_list = [] #... und die Target (Label) Liste, damit wir die Bilder mit Labels versehen können
train_data=[]  #wird zu einer Liste aus batches
files = listdir(imagesdirpath) #hier habe ich anfangs den Fehler gemacht, das zip-file nicht zu entpacken, daher hat es den Pfad nicht erkannt
for i in files:
    if ".ini" in i: files.remove(i) #Da ich wegen dem Augmentor ein paar Mal eine Fehlermeldung bekommen habe, weil er ein Unbekanntes "Bild" .ini auch erzeugt hatte
if path.exists(modelpath) is False: #das Modell kreieren wir nur, wenn es noch nicht existiert:
    for idx, i in enumerate(range(len(files)-1), start=1):  #liefert mir Liste mit allen Dateinamen zurück
        f = random.choice(files) # liefert zufälliges element aus der files Liste -- wenn wir nur das haben, kann es passieren, dass wir immer dasselbe Element bekommen. Darum:
        files.remove(f) # müssen f aus den files removen, nachdem es schon gewählt wurde
        img = Image.open(imagesdirpath + "/" + f).convert('RGB') # wir öffnen das gewählte Bild und speichern es als array
        img = transforms.functional.to_grayscale(img) #Grayscale wegen RAM
        img_tensor = transform(img) #das Bild in einen Tensor transformieren, damit nn damit arbeiten kann (Tensoren sind eine Verallgemeinerung von Matrizen und werden durch n-dimensionale arrays repräsentiert)
        train_data_list.append(img_tensor) #nachdem das Bild jetzt prepariert ist, wird es in die Trainingsliste hinzugefügt
        HatRiss = 1 if 'NG' in f else 0  #jetzt definieren wir die Targets: Die Abbilddung "Hatriss" gibt 1 aus, wenn im Bildnamen NG steht, sonst 0
        HatkeinenRiss = 1 if 'OK' in f else 0 #das zweite Target: die Abbildung "HatkeinenRisss" macht genau das Umgekehrte
        target = [HatRiss, HatkeinenRiss] #das soll dann also der Output (besteht also aus 2 Neuronen)/das Target sein.
        print(target) #zeigt entweder [0,1] oder [1,0] an, ersteres wenn das target keinen Riss hat
        target_list.append(target) #in unsere Traget Liste wird jetzt, nachdem es benannt wurde, das Traget (also die Auswertung vom random gewählten Bild) hinzugefügt
        print("Image " + str(idx) + " out of " + str(len(listdir(imagesdirpath))-2) + " loaded!")
        if len(train_data_list) >= (len(listdir(imagesdirpath))-2) : #Wenn in die erstellte Trainingsdatenliste größer gleich so viele Bilder hinzugefügt wurden, wie in dem von mir "eingefütterten" Trainingsbildordner" drinnen sind 
            print("Appending data to a single list...") #dann fügen wir alles in einer "Endliste" zusammen:
            train_data.append((torch.stack(train_data_list), target_list)) #stackt jetzt die Trainingsdatenliste und die Liste mit den Auswertungen zusammen =(Trainingsdaten, Liste) <- das hier ist dann ein batch, da sind Bilder mit Lösungen drinnen
            train_data_list = [] #leert die Liste, die aus den Tensoren der ursprünglichen Trainingsbilder entsteht, um RAM Speicherplatz zu entlasten (die Liste wird ja jetzt nicht mehr benötigt, da sie in der gerade defninerten Liste enthalten ist)
            print("Appending finished!")
            
#Jetzt kann das Neuronale Netz definiert werden:
class Netz(nn.Module):
    def __init__(self): #initializer Methode, initialisiert/kreiert memory space, um sich an Details über die neuen Elemente, die dann in dieser Klasse definiert werden, erinnern zu können
        #self wird jetzt per default als ertses Argument genommen
        super(Netz, self).__init__() #jetzt rufe ich ich die neu definierte Klasse __init__ auf (calling the class)
        self.conv1 = nn.Conv2d(1, 2, kernel_size=5) #wird zur ersten Variable, wenn man ein Element (=instance) der__init__(self) class definiert
        self.conv2 = nn.Conv2d(2, 4, kernel_size=5) #Da wir mit Bilder arbeiten kommen mal drei convolutional Schichten gleich als nächste Argumente in den initializer rein
        self.conv3 = nn.Conv2d(4, 8, kernel_size=5) #Zahlen? = Dimensionen der Matrizen, die den Filter darstellen
        self.fc1 = nn.Linear(7616, 1000) #nimmt convolution Output und gibt eine Klassifizierungsentscheidung ab
        # = fully connected layer soll als input genau die Anzahl an Neuronen haben, die torch.Size(, ., ., .) => . mal . mal .  ausspuckt
        self.fc2 = nn.Linear(1000, 2)
        
    def forward(self, x): ##
        x = self.conv1(x)
        x = F.relu(x) #Aktivierungsfunktion, relu f(x) = mas(0,x), wobei x der input ist
        x = F.max_pool2d(x,2) #maxpooling teilt das ganze Bild (welches aus unserem Convolutional Layer rauskommt) 
        #in 2x2 Pixelblöcke und nimmmt als output von jedem Block das Maximum (=> verkleinert das Bild)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2)    
        x = x.view(-1, 7616) ##    -1, da der erste Wert, also die batch size, uns nicht interessiert
        x = F.relu(self.fc1(x))
        x = F.dropout(x) #nimmt einen random Teil der Activation und zerstört diesen. 
        #So wird das nn gezwungen eine robustere Wiedererkennung zu erlernen und sich auf mehrere Eigenschaften des inputs gleichzeitig zu konzentrieren, anstatt immer auf dieselbe(die falsch sein könnte)
        x = self.fc2(x)
        return F.softmax(x, dim=1) #softmax nimmt die letztendliche Klassifizierungsentscheidung aus dem fully connected layer und wandelt diese in Wkeiten um (mithilfe der Eulerfunktion)
        #jetzt müssen wir in unseren fully connected layer umsteigen, aber wir wissen ja nicht, wie unsere Daten in dem Fall jetzt aussehen
        exit()
    
        
#jetzt generiere ich mein model als neues Netz (Instance definieren):
model = Netz()


optimizer = optim.RMSprop(model.parameters(), lr = 0.001) #Festsetzung der Learning rate

def train(epoch): #jetzt wird definiert, was im Training passieren soll:
    model.train() #model auf train setzen, damit es auch wirklich trainiert wird und nicht nur ausgeführt
    for idx, (data, target) in enumerate(train_data): #wenn der Index vom gewählten Bild in der Nummerierung der in Zeile 90 (train_data.append) definierten Batchliste ist, dann:
        target = torch.Tensor(target) #wird wieder einen Tensor draus gemacht, damit das nn damit arbeiten kann
        data = Variable(data) ##1.Batch entry: data (also das Bild) wird als Variable genommen
        target = Variable(target) #2.Batch entry: data und target müssen ja immer noch als Variablen verarbeitet werden
        optimizer.zero_grad() ## Gradient descent
        out = model(data) #output soll der output vom definierten Modell (in der Klasse Netz) sein, wenn man data reinfüttert
        criterion = F.binary_cross_entropy #das Kriterium ist,dass die Bilder ja in zwei Klassen unterteilt werden sollen
        loss = criterion(out, target) ## Loss soll als criterion von output und target berechnet werden
        loss.backward() ## Loss soll back - propagatet werden
        optimizer.step() # wollen unseren Optimizer sagen, dass er etwas tun soll
        
## Optimizer zero grad sucht die stelle, an der der Gradient = 0 ist?

epoch_num = range(0,int(epochs))

if path.exists(modelpath) == False: #man schaut, ob unsere Model schon trainiert ist, dh ob es existiert
    for idx, epoch in enumerate(epoch_num, start=1): #man will 30 epcohen lang trainieren, falls es noch nicht existiert
        print("Epoch " + str(idx) + " out of " + str(max(epoch_num)+1) + " started!")
        train(epoch) #und wendet dann die Funktion train an, in der das Model trainiert wird
        print("Epoch " + str(idx) + " out of " + str(max(epoch_num)+1) + " finished!")
    torch.save(model, modelpath) #dann wird das trainierte Modell gespeichert
else: #sonst, falls das Model existiert,dann:
    model = torch.load(modelpath) #wird das Modell geladen
    model.eval() #.. evaluiert
    vals = [] #dann wird ein Werte set erstellt
    tests = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] #...die Testsetindices (das testset beinhaltet 10 Bilder; nummeriert von 1-10)
    result = [] #und noch ein Lösungsset wird erstellt
    if timagesdirpath is None: sys.exit() #falls dann kein testimageset existiert, soll das system verlassen werden
    else: #sonst soll geprüft werden, ob man das Modell nützen kann
        for i in listdir(timagesdirpath): # also für die Bilder aus dem Testset:
            if ".ini" in i : continue #wie oben beschrieben: wenn .ini wieder drinnen ist dann ignorieren
            test_img = Image.open(timagesdirpath + "/" + i) #für die anderen, richtigen Formate werden die Bilder geöffnet
            test_img = transforms.functional.to_grayscale(test_img) #... dann in Grayscale transformiert wegen RAM
            test_img_tensor = transform(test_img) #mit der vorher definierten Transform-Funktion in das richtige Format transformiert (wird unter Anderem zu einem Tensor)
            loading_image = model(test_img_tensor[None, ...]) #... dann wird der Tensor zu unserem Model geschickt und dieses spuckt  
            cpu_pred = loading_image.cpu() #und zusammen damit an den cpu-Prozessor
            numpyresult = cpu_pred.data.numpy() #dort wird es zu einem array formatiert
            listresult = np.ndarray.tolist(numpyresult[0]) ##formatiert den n-dim array in eine Liste um, 0 bedeutet
            endresult = str(listresult).split(", ") ##splitet den string der Liste durch Beistriche
            bad = endresult[0].replace("[", "") #ob das Cookie "bad"/gebrochen ist, entscheidet der erste Wert (der Index 0 im endresult hat), siehe Zeile 82
            good = endresult[1].replace("]", "") #ob das Cookie "good" ist, entscheidet der zweite Wert (der Index 1 hat), siehe Zeile 83
            if good > bad: vals.append(1) #wenn das Modell eher glaubt, dass das Cookie gut ist, dann füge 1 zu den Ausgabewerten (=Liste "vals") hinzu
            else: vals.append(0) # sonst (wenn das Modell eher glaubt, dass es gebrochen ist), füge 0 hinzu 
            #(diese Liste "vals" wird später ausgegeben, um zu sehen wie gut das Modell ist)
        for l, f in zip(vals,listdir(timagesdirpath)): #wenn l in der Liste vals ist, und f ein Testimage, dann:
            if ".ini" in f: continue #wie immer: wenn f ini beinhält ignorieren, sonst:
            if int(f[:2].replace(".", "")) in [1, 2, 3, 4, 5]: ##löschen den Punkt aus dem Namen von f und prüfen, ob es ein Cookiebild mit index 1-5 ist 
                #DIe Bilder 1-5 im Testset sind Cookies, die schlechten Cookies
                if l == 0: result.append(1) #wenn also der Wert für l aus der "vals" Liste dann auch noch 0 ist, dann bedeutet das, dass 
                #good<bad, also dass das Cookie wsl schlecht ist und das Bild somit richtig predicted wurde
                else: result.append(0) #sonst wurde es schlecht predicted (falsch)
            if int(f[:2].replace(".", "")) in [6, 7, 8, 9, 10]: #die Bilder 6-10 sind von guten Cookies, also:
                if l == 1: result.append(1) #wenn auch der Wert l als 1 (good>bad) predicted wurde, dann lag das Modell richtig, also soll 1 ausgegeben werden
                else: result.append(0) #sonst lag es falsch und 0 soll ausgegeben werden
        plt.scatter(tests, result,  color='black') #jetzt wird ein scatter gebildet, der tests set in Bezug zu result bringt
        plt.plot(tests, result, color='blue', linewidth=3) #.. damit wird ein Plot gezeichnet mit x=tests und y=result Set
        plt.show() #... und der Plot wird ausgegeben, um zu zeigen, wie gut das Modell war
        
