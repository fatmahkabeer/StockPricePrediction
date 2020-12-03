
# I get this dataset from matplotlibwhich is about Historical DCE futures prices: Iron Ore.
# Here I am trying to learn how to predict the future close price. 
#https://www.quandl.com/data/DCE/IX2020-Iron-Ore-Futures-November-2020-IX2020

#%% import libraries

from DiscoveringData import DiscData
from PrepareData import PreparingData

from MLModeling import LinearRegressionModel
from MLModeling import KNearestNeighboursModel
#I can not push to git hub while useing this package. 
from MLModeling import MARSmodel
from MLModeling import RandomforestModel
from MLModeling import XGBoostModel


import pandas as pd 
import numpy as np

#for visualizing
import matplotlib.pyplot as plt


#%%read data
df = pd.read_csv('DCE-IX2020.csv')

#%%
#setting index as date values
df.index = df['Date']
       
#sorting
df = df.sort_index(ascending=True, axis=0)


#split the dataframe into train and validation sets to verify our predictions
#we cannot use random splitting since that will destroy the time component. So here we have set the last year’s data into validation and the 4 years’ data before that into train set.       
train = df[:200]
valid = df[200:]

#%% Split data:
x_train = train.drop('Close', axis=1)
y_train = train['Close']

x_valid = valid.drop('Close', axis=1)
y_valid = valid['Close']

# %% Discovering Data:

# Create an object of DiscData class inside DiscoveringData.py file.
dis = DiscData(df)

#%%
#To plot the relationships between columns:
#This diagram helps us try to figure out is there some kind of relationship between variables:
#Aa the graph illustrets, the relationships between columns almost linear unless with the last three columns: #This diagram helps us try to figure out is there some kind of relationship between variables: Volume, Open, and Turnover
dis.plotRelation()


#To find missing values:
#There are not much missing values in this dataset. 
dis.isNan()

#Data shape:
dis.dataShape()

#Distrubution of all Varabiles:
#As it showing in histograms, There is skew in columns distribution. 
dis.plotDataHist()

#Distrubution of Close price:
dis.plotCloseHist()


# %% Preapering data 

#Create an object of PreparingData class from PrepareDate.py file
#prep = PreparingData(df) 


#fill all missing values.
#y_train = prep.fillNa(y_train)


#removing features with low variance
#prep.RmLowVar()



#removing correlation:
#prep.Correlation()


#%% Modleing 
#Create an object of LinearRegressionModel class from Modeling.py file
linearModel = LinearRegressionModel(df)

linearModel.linearAccuracy(x_train, y_train, x_valid, y_valid)
# Applying Linear Regrassion gives me a root mean square deviation = 41

#This is how the plot looks like:
linearModel.linearPlot(train, valid)
#%%
#Create an object of KNearestNeighboursModel class from Modeling.py file
knnModel = KNearestNeighboursModel(df)

knnModel.knnAccuracy(x_train, y_train, x_valid, y_valid)
#Applying K Nearest Neighbours gives me a root mean square deviation = 326 which gives us a significant error in general and in comparison to the Linear Regression. 

#This is how the plot looks like:
knnModel.knnPlot(train, valid)

#%%
#Create an object of MARSmodel class from Modeling.py file
marsModel = MARSmodel(df)

marsModel.marsAccuracy(x_train, y_train, x_valid, y_valid)
# Applying MARS gives me a root mean square deviation = 42 which is kind of similar to Linear Regression and much improvement in comparison to K Nearest Neighbours.

#This is how the plot looks like:
marsModel.marsPlot(train, valid)

#%%
#Create an object of RandomforestModel class from Modeling.py file
randomfModel = RandomforestModel(df)

randomfModel.randomfAccuracy(x_train, y_train, x_valid, y_valid)
# Applying Random Forest gives me a root mean square deviation = 95.8. It's better than wK Nearest Neighbours.
# However it still worest than Linear Regression and MARS.

#This is how the plot looks like:
randomfModel.randomfPlot(train, valid)


#%%
#Create an object of XGBoostModel class from Modeling.py file
xgboostModel = XGBoostModel(df)

xgboostModel.xgboostAccuracy(x_train, y_train, x_valid, y_valid)
# Applying XGBoost is the best with root mean square deviation = 31.7.

#This is how the plot looks like:
xgboostModel.xgboostPlot(train, valid)

