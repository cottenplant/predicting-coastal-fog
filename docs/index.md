
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
Coastal fog is a common occurrence in many areas of California. "June Gloom", as it is popularly known to the residents of Los Angeles County, is a paradoxical weather event where baking hot inland conditions give rise to an inversion layer, where rising heat creates a vacuum which draws in and traprs ocean air along the coastal regions, withe the temperatre variatiom precipating dissolved water vapor in the air to create a dennse fog hugging the coastline. Named for the month it tends to occur most dramatically, this phenomenon sees temperatures near the beach plummet in the late afternoon and evening, a reveral from hot summer weater which may intrude eastward into the greater Los Angeles and downtown LA areas, a pattern which generally holds until mid-morning the followind day when it burns off, only to yield to cooler inversion layer in the late afternoon.. This cycle may persist for days or weeks, with some Sotuhern Califronia Sumers seeing nothing but a majority of "gloomy" days.

## Objective
Presented is an analysis of weather conditions and trends both at Santa Monica Airport (KSMO) and Los Angeles International Airpot (KLAX) which, combined, give rise to this coastal marine layer. Long-term trends due to  climate global warming and shifts in seasonal temperatures independant  corraborate lcoal finding; such as those related to climate change are also briefly touched on. Once some relevant features of the data are identified, machine learning techniques (i.e. classifiers) will be applied to determine the best model for predicting fog.

# Methods
Datasets were acquired under an open license. KSMO data was downloaded via the Wunderground.com API, and KLAX data was acquired through the NOAA portal online. I haven't published any actual data to avoid any possible issues. Dataset consumption and visualization were accomplished using numpy, pandas, matplotlib, and seaborn, whereas machine learning algorithms were deployed within scikit-learn.

# Analyzing data and features
## trends
![Alt](https://user-images.githubusercontent.com/25386879/45439620-26009e80-b66f-11e8-9c2a-fa0f7179cada.png)
From the (above) 3d graphs, one can easily discern wind direction, speed and dew point relationships during the period of measurement : 1972 - 2018. Such a comprehenesive representative figure summarizes succinctly the relationshio\p betwewen wind directionn, speed and humidity. Not the 
Grouping the trade winds (ocean breezes rich in water content an higher dew point are easiy distinguishable from the so-called "Santa Ana" or earsterly winds that occur in Fall and Winter, patterns of persistent high pressure which bring unusually dry and windy deserst air from the valleys to th East of Los Anglees.

## features
Stirpping down the many features into pricipal components involved eliminating derivative features (i.e. feature which we just conversions or derivations from raw observed data. We also edited for instance a column in the NOAA text which listed Fog, Rain, Snow, Hail, Thunderstorm and Tornado in a binary format to reflect real values for analyis. We selected the most relative and complete features for us in a gamute of machine leaarnning classifiers containd withing the scikit-learn ecosysteeyin

## cleaning the data
Tha data required minimal editing for clarity and dropping of some tagential values, many of whic where incomplete. In the present analysis, we use features with a direct relevance to out hypothesis, namely: how can we predict the occurrence of fog based on certain pre=existing factors.

# Methods, packages and toolkits
numpy, pandas, saeaborn, plot.ly, scikit-learn and other played an integral roll in the consumption an leverage of this necessary data set

# Data modeling
## train/test
We rain a grid search to find and rate the various classifiers for parameters of model efficiency

## compare scikit-learn classifiers
I ran a test script deploying all sci-kit learn classifiers to determine the ideal combination of recall, precision and occasionally atmospheres. The n number of examples of foggy days was far lower than total daily observations

## select – precision, accuracy, recall
Interesting the gradient boosting model returned the higest accuracy but since our data contain relatively much fewer classified fog events, some sampling error certianly skewered prediction results.

# Publishing data
Knowing when fog rolls in, which is frequently a sudden and disappointing event, perhaps to reduce traffic of people heading to the beach as well as provide fire fighters wih cutting endge information. 

## figures
Various figures of wind patterns plotted agains wind speed have been synthesized and uploaded in the /figures/ folder.
Please check soon for more updates!

## fog-predict tool
## FogBot announcements

#Conclusions and future directions

