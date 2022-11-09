# Machine-Vision
The goal of the project was to train a neural network to recognize whether or not a Waffle is broken.

First, I fix the name of the model. If the model name is non-existent, then a new model with the provided name is created automatically.
Then I fix the training set and the test set, which I chose to be the set of artificially cretaed pictures (Code see "augmented" file), as well as the number of training sessions.
Next, the used data needs to be normalized. A new (empty) list for the training-data is created (as a list of batches) as well as a target list, such that the pictured can be labelled. 
I fix the directory and make sure that there is no unnecessary ".ini" file in it (due to the augmentor that I used). 
files = listdir(imagesdirpath) #hier habe ich anfangs den Fehler gemacht, das zip-file nicht zu entpacken, daher hat es den Pfad nicht erkannt.
If the model does not already exist, then a model has to be created. It should obtain a list with all files, take a random element out of it (remove it from the list such that it is not used again), open this element (file), transform it to the grayscale and then to a tensor (to make it possible for the neural network to work with it). Subsequently the obtianed tensor is added to the training list.

Now it needs to be fixed how the trainingset should be iterpreted from the model: I put a "NG" in the filename when the Waffle is not broken, in this case it should be assigned the number 1 for "HatRiss", as this is true in this case. If it is assigned the number 0 for "HatRiss" then is is not broken and has the letters "OK" in the filename. Analogeously this will hold for "HatkeinenRiss". The output (target) will be then appended to the list of "loaded images". Afterwards everything will be concluded into a final list, where the training data list is stacked to the list of outputs. So each image is assigned its output, in form of a "batch".

Now that the training batchset is prepared, it can be used to train the CNN, which is defined in th following:

First, a memoryspace is created/initialized in order to be able to recall the details about the elements, which are defined in this class. Self is now used as the first element per default. After calling the class, the first three factors that come into the initializer are convolutional layers. The parameters that this function needs are the dimensions of the matrices, which represent the filter. Afterwards, the classification type is defined with a fully connected layer. Its input should be the number of neurons, which are the output of torch.Size(, ., ., .) => . times  . times . Afterwards, another fully connected layer is added.

In the next steps, the relu (activation function) is used. The maxpooling in 2d, which is used afterwards is a technique to minimize the number of pixels (it takes each set of 2x2 pixels and outputs the maximum of each "block" of pixels. This process is repeated three times. The activation function is used again and afterwards the dropoutfunction, which helps to make the process/image processing more robust by destroying a random part in the activation. In that way, it has to pay attention to more than one key property instead of just one (which could be misleading). 

The following softmaxfunction takes the last decision about the classification from the fully connected layer and transforms them into probabilities (with an Euler function).
Then the model is created as a new network.

The learning rate is defined (can be interpreted as the "step size").

Then the training process is defined: the above defined model is set to trainmode. 


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
        
