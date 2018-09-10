## Summary
FogBot is a small tool predicting the occurrence of fog in a coastal region of Los Angeles, California, USA. After running the gamut of classifiers available in the scikit-learn machine learning package, the Decision Tree Classifer performed best in terms of precision (0.26) -- other models achieved higher 'accuracy' but had lower rates of true positives compared to total positives. This is likely due to an asymmetrical dataset; of the approximately 5400 daily measurements from Santa Monica Airport, fog occurred only on around 300 of these days.

The tool will be accessible via [direct chatbot interaction](https://telegram.me/fog_check_bot) using Telegram Messenger. I will also develop an embedded version for this site. Future support for other coastal cities will be added as well. 

## Background
"June Gloom" is a familiar phenomenon in many parts of California. Named for the month it tends to appear the most frequently, the characteristic fog or marine layer occurs due to temperature differences accross the cold Pacific ocean and typically baking inland temperatures. This somewhat paradoxical weather phenomenon leads many in coastal areas to feel a bit down in the heat of summer. Temperatures plummet and the sky darkens quickly as the fog rolls in. The very low ceiling (altitude of clouds off the ground) of the marine layer contributes to this 'under the weather' feeling.

According to Wikipedia.org, fog forms when the differnce between temperature and dew point is less than 2.5 degrees Celsius. The international definition of fog is a visibility of less than 1 kilometre (3,300 ft); mist is a visibility of between 1 kilometre (0.62 mi) and 2 kilometres (1.2 mi) and haze from 2 kilometres (1.2 mi) to 5 kilometres (3.1 mi). (wikipedia.org)

In this analysis, fog events are classified according to the agencies recording the data. For instance, the NOAA data set has a column for all inclement weather (Fog, Rain, Snow, Hurricane, Thunderstorm, Tornado), while WeatherUnderground historical summaries obtained via the wunderground API report under a dedicated 'fog' column. I've included jupyter notebooks of my thought process in the data cleaning and feature engineering for transparency.

## Objective
In this analysis, we explore various aspects of weather measurements and trends both at Santa Monica Airport (KSMO) and at Los Angeles International Airpot (KLAX) to tease out the factors leading to marine layer, as well as touch on long-term trends in climate such as those related to climate change.

## Methods
KSMO data was downloaded via the Wunderground.com API, and KLAX data was acquired through the NOAA portal online. Datasets are used under an open license - I haven't published any actual data to avoid any possible licensing issues.

Dataset consumption and visualization were accomplished using numpy, pandas, matplotlib, and seaborn data science packages, while machine learning algorithms were deployed within the scikit-learn ecosystem.

Check back soon for updates to this site!

