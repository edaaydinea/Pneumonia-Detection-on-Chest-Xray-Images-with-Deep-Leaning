# Pneumonia Detestion on Chest X-ray Images with Deep Learning (Keras)
This repository includes the slides and coding parts for this project.

This project is carried out by *Eda AYDIN, Nesibe Betül DÖNER, Okan TOPAL, Berkay DİRİL* under the supervision of *Engin Deniz ALPMAN* in the **[Data Science for the Public Good program](https://www.kodluyoruz.org/bootcamp/data-science-for-the-public-good-istanbul-ankara/)**

The dataset of this project is obtained from the [Kaggle - Chest X-ray Images(Pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) 

Note: The data sets to be used in the project comply with the health-ethical rules and are suitable for use as a license.


## A. BUSINESS UNDERSTANDING

### Pneumonia Challenges to WaitinG Solve

- Lungs are the most affected organs of COVID-19
- Early Detection for high results in the treatment of the disease

### Our Goal
- Pinpoint detection of pneumonia disease from chest X-ray images

### Context
![Example of Chest X-rays in patients](https://i.imgur.com/jZqpV51.png)

The normal chest X-ray(left panel) depicts clear lungs without any areas of abnormal opacification in the image.
Bacterical pneumonia (middle panel) typically exhibits a focal lobar consolidation, in this case in the right upper lobe (white arrows), whereas viral pneumonia(right panel) manifets with a more diffuse "interstitial" pattern in both lungs.



## B. DATA UNDERSTANDING

### Collectiong of the Raw Data

The dataset is organized into 3 folders (train, test, val) and contains subfolders of each image category (Pneumonia / Normal). There are 5,863 X-Ray images (JPEG) and 2 categories(Pneumonia/Normal)

Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.

For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.

### Changing Environment
- Used Tensorflow to change environment from CPU to GPU for working fastly in modelling part.

### Data Loading
- Directories were taken from different folders for each image files to be used in the visualization, augmentation and modeling phases. 

### Labelling
- 0 and 1 labels were added to work in the modelling phase to the pictures determined as pneumonia and normal.

### Data Visualization

**Normal / Pneumonia Image Visualization fro Train Dataset**

![normal-pneumonia-images](https://github.com/edaaydinea/Pneumonia-Detection-on-Chest-Xray-Images-with-Deep-Leaning/blob/main/normal-pneumonia%20image%20visualization.png)

![distribution-with-bar](https://github.com/edaaydinea/Pneumonia-Detection-on-Chest-Xray-Images-with-Deep-Leaning/blob/main/data%20distribution%20with%20bar.png)

![distribution-with-pie](https://github.com/edaaydinea/Pneumonia-Detection-on-Chest-Xray-Images-with-Deep-Leaning/blob/main/data%20distribution%20with%20pie.png)
### Data Preparation

#### Changing the each image size to prefered size
- Changed the each image size to 150x150

#### Grayscale Normalization
- A grayscale normalization was performed to reduce the effect of lighting differences.
- The data were normalized from 0–1 by dividing the train, validation, and test data by 255. 

#### Reshaping the data for deep learning model
- Each image size was reshaped from -1 to 1 to use the images as NumPy arrays in the deep learning model.


## C. DATA AUGMENTATION

![data-augmentation](https://github.com/edaaydinea/Pneumonia-Detection-on-Chest-Xray-Images-with-Deep-Leaning/blob/main/data%20augmentation.png)

- Techniques applied as data augmentation for train dataset:
   - Randomly rotate some training images by 30 degrees
   - Randomly Zoom by 20% some training image
   - Randomly shift images horizontally by 10% of the width
   - Randomly shift images vertically by 10% of the height
   - Randomly flip image horizontally

## D. DATA MODELLING

- The CNN model was built by using Keras and the accuracy result on test data is : 95.27%
- ReduceLRonPlateau was used in the model to reduce the learning rate when the model stopped improving the metric.

## E. EVALUATION

**Accuracy and Loss Graphs**
![accuracy-loss-graphs[(https://github.com/edaaydinea/Pneumonia-Detection-on-Chest-Xray-Images-with-Deep-Leaning/blob/main/evalaution.png)

**Some of the Correctly Predicted Classes**

![correct-image](https://github.com/edaaydinea/Pneumonia-Detection-on-Chest-Xray-Images-with-Deep-Leaning/blob/main/correct-images.png)

**Some of the Inccorectly Predicted Classes**

![false-images](https://github.com/edaaydinea/Pneumonia-Detection-on-Chest-Xray-Images-with-Deep-Leaning/blob/main/false-images.png)

## G. RESOURCES

**Notebook Resources**

  * https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
  * https://www.kaggle.com/aakashnain/beating-everything-with-depthwise-convolution
  * https://www.kaggle.com/amyjang/tensorflow-pneumonia-classification-on-x-rays
  * https://www.kaggle.com/madz2000/pneumonia-detection-using-cnn-92-6-accuracy
  * https://www.kaggle.com/sanwal092/intro-to-cnn-using-keras-to-predict-pneumonia
  * https://www.kaggle.com/homayoonkhadivi/medical-diagnosis-with-cnn-transfer-learning
  * https://www.kaggle.com/arbazkhan971/pneumonia-detection-using-cnn-96-accuracy
  * https://jovian.ai/edaaydinea/applying-cnn-on-chest-xray-images

**Website Resources**

* ***Tensorflow - Keras Documentations***
    * https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
    * https://keras.io/api/layers/convolution_layers/

* ***Others***
    * https://towardsdatascience.com/deep-learning-in-healthcare-x-ray-imaging-part-5-data-augmentation-and-image-normalization-1ead1c02cfe3

* **Book Resouces**
    * Hands-on Machine Learning with Scikit-Learn, Keras & Tensorflow
    * Deep Learning with Python
    * Tensorflow for Deep Learning
    * Programming Computer Vision with Python
