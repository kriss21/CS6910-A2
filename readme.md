# CS6910-A2
## CS18B021 - CS18B023

---

## Part A

The *iNaturalist* dataset contains images in the train and test folders. We need images for validation also which we take out from the training set of images. To keep this data consistent over different model runs, we divide the train set into train and val sets before hand using the `pip` library `split folders`. We use this library and separate out 0.1 fraction of train data into val set. The splitted data can be found [here](https://drive.google.com/drive/folders/12cKzbU1x1Umo5lmhK-FkpZz83VzsWUjo?usp=sharing).

**partA-q2.py** :
This file contains the code for part A question 2 of the assignment. The file has code for model building first, then it runs the wandb sweep over different hyperparameter configurations.
The best hyperparameter values are then noted down from wandb and the best model is trained using them by running the juypyter notebook *best-model-train.ipynb*.

**best-model-train.ipynb** : 
This is a jupyter notebook that trains the best model found using the wandb sweep observations. The best model is then saved in the file `best_model.h5`.

**partA-q4.ipynb**:
This is a jupyter notebook that contains the code for part A- question 4 of the assignment. The file first tests the best model found `best_model.h5` on the test data. The file then plots the prediction of this model on several images from the test data. It then plots the filters in the first convolution layer on passing an image from the test data.

**partA-q5.ipynb** :
This is a jupyter notebook containing the code for part A question 5, guided backpropogation, of the assignment. The code runs guided backpropation on 10 random neurons in the last convolution layer of the model on an image from test dataset.

**Note** : The data directory location is relative to our system directory structure. To reproduce the results set the data directory correctly to your data location while loading data.