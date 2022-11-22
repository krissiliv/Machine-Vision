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

Then the training process is defined: the above defined model is set to trainmode. Each picture is transformed into a tensor. The data and the target are both taken as a variable. The optimizer "zero gard(ient)" is used.  The output will be the data which "ran" through the above defined model. And the loss should be calculated as the criterion of output and target. It should be back-propagated (back propagation of loss) and then the optimizer should take a step (it should act).

The epoch number (number of the training-process) is defined as "epoch_num".

Then it is checked, whether or not the model path already exists (if it was already trained). If yes, the existing model path is used, if not, a new model path is created. In each epoch, the model is trained and afterwards it is saved (the model path is created). If the model path does exist, the model is loaded, evaluated.
If the trainingset is not found, the system is exited.
If it exists, the testpictures are opened one by one, transformed into a greyscale picture due to the RAM (colour is not relevant here), then the transform function is used to transform the image into the right format. The image gets loaded, sent to the cpu-processor, re-formated into an array, then makes a n-dimensional list out of it, which gets splitted by commas. In the endresult, the first number tells us if the cookie is bad and the second one tells us if the cookie is good. If good > bad (this means the "good" entry is 1 and the "bad" entry is 0) then a "1" is added to the values. On the other hand, if the model believes that the cookie is bad (broken) then a "0" is added. This list of values is later used to check the model performance.

We take this list and the testimages (where the images numbered 1-5 show broken (bad) cookies and the rest show good ones). 
If one of these images with number 1-5 is assined the number 1, which is the representant for bad cookies, then the model has predicted the image correctly. In this case "1" is added to the results, else (if the prediction was sincorrect) "0" is added to the results. 
The analogeous thing happens to the testimages with numbers 6-10 (the good cookies).

Finally the performance of the model is plotted in order to be able to decide how good it is. The x-axis represents the test-images and the y-axis represents the outcome (whether the prediction was wrong or correct).

_______________________________________________________________________________________________________________________________________
How can this Code be used? A step-by step guide.

---The setup ---
Create a folder, where you save this Code-file. You can name it as you like but for referring reasons and to make the below describtion easier I name it "Folder1".

In Folder1, create another folder named "Test_AUGMENTED". This will be the test-images. It should contain 10 images, numbered from 1-10. The pictures 1-5 should have broken objects on them. The pictures with number 6-10 should show not-broken objects.

Also in Folder one, create a folder named "Train" where you save a good mix between broken and not-broken objects. The more the better.

Then you can open the Code and in line 25, you can choose which name your model should have.


        
