
```
===CLASSIFIER===SUMMARY===
n = 1000
                              Accuracy  Precision    Recall
Algorithm                                                  
Decision Tree                 0.909068   0.195421  0.224249
Gaussian NB                   0.934721   0.132978  0.038846
Gradient Boosting Classifier  0.945638   0.467500  0.097002
K-Nearest Neighbors           0.942225   0.301554  0.059547
Linear SVM                    0.946546   0.000000  0.000000
Logistic Regression           0.946282   0.012771  0.000329
Random Forest Classifier      0.943096   0.366783  0.087023
rbf SVM                       0.946498   0.000000  0.000000
```

# Background
Coastal fog is a common occurrence in many areas of California. "June Gloom", as it is popularly known to the residents of Los Angeles County, is a paradoxical weather event where baking hot inland conditions give rise to an inversion layer, drawing in and trapping damp coastal air along the beach regions. Named for the month it tends to occur most dramatically, this phenomenon sees coastal temperatures plummet in the late afternoon and evening, a pattern which generally holds until mid-morning the following day. This cycle may persist for days or weeks, with some summers seeing a majority of "gloomy" days.

## Objective
Presented is an analysis of weather conditions and trends both at Santa Monica Airport (KSMO) and at Los Angeles International Airpot (KLAX) that give rise to this coastal marine layer, as well as touch on long-term trends in climate such as those related to climate change. Once some relevant features of the data are identified, machine learning techniques (i.e. classifiers) will be applied to determine the best model for predicting fog.

# Methods
Datasets were acquired under an open license. KSMO data was downloaded via the Wunderground.com API, and KLAX data was acquired through the NOAA portal online. I haven't published any actual data to avoid any possible issues. Dataset consumption and visualization were accomplished using numpy, pandas, matplotlib, and seaborn, whereas machine learning algorithms were deployed within scikit-learn.

# Analyzing data and features
## trends

## features
## cleaning the data

# Methods, packages and toolkits


# Data modeling
## train/test
## compare scikit-learn classifiers
## select – precision, accuracy, recall

# Publishing data
## figures
## fog-predict tool
## FogBot announcements

#Conclusions and future directions

