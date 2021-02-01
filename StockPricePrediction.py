
# I get this dataset from matplotlibwhich is about Historical DCE futures prices: Iron Ore.
# Here I am trying to learn how to predict the future close price. 
#https://www.quandl.com/data/DCE/IX2020-Iron-Ore-Futures-November-2020-IX2020

# %% import libraries
import pandas as pd 
import numpy as np

#for visualizing
import matplotlib.pyplot as plt

from DiscoveringData import DiscData
from PrepareData import PreparingData

from MLModeling import LinearRegressionModel
from MLModeling import KNearestNeighboursModel
#I can not push to git hub while useing this package. 
from MLModeling import MARSmodel
from MLModeling import RandomforestModel
from MLModeling import XGBoostModel
from MLModeling import ARIMAModel

#LSTM
from DLModeling import LSTMModel




# %%read data
df = pd.read_csv('DCE-IX2020.csv')

# %%
#setting index as date values
df.index = df['Date']
       
#sorting
df = df.sort_index(ascending=True, axis=0)

#creating a separate dataset
newData = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])

for i in range(0,len(df)):
    newData['Date'][i] = df['Date'][i]
    newData['Close'][i] = df['Close'][i]

newData['Date'] = pd.to_datetime(newData['Date'])
newData.set_index('Date', inplace = True)


#split the dataframe into train and validation sets to verify our predictions
#we cannot use random splitting since that will destroy the time component.       
train = newData[:200]
valid = newData[200:]



# %% Discovering Data:

# Create an object of DiscData class inside DiscoveringData.py file.
dis = DiscData(df)

# %%
#To plot the relationships between columns:
#This diagram illustrets, the relationships between columns.
#It almost linear unless with the last three columns: Volume, Open, and Turnover
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


# %% Modleing 
#Create an object of LinearRegressionModel class from Modeling.py file
linearModel = LinearRegressionModel(df)

linearModel.linearAccuracy()
# Applying Linear Regrassion gives me a root mean square deviation = 41

#This is how the plot looks like:
linearModel.linearPlot()
# %%
#Create an object of KNearestNeighboursModel class from Modeling.py file
knnModel = KNearestNeighboursModel(df)

knnModel.knnAccuracy()
#Applying K Nearest Neighbours gives me a root mean square deviation = 326 which gives us a significant error in general and in comparison to the Linear Regression. 

#This is how the plot looks like:
knnModel.knnPlot()

# %%
#Create an object of MARSmodel class from Modeling.py file
marsModel = MARSmodel(df)

marsModel.marsAccuracy()
# Applying MARS gives me a root mean square deviation = 42 which is kind of similar to Linear Regression and much improvement in comparison to K Nearest Neighbours.

#This is how the plot looks like:
marsModel.marsPlot()

# %%
#Create an object of RandomforestModel class from Modeling.py file
randomfModel = RandomforestModel(df)

randomfModel.randomfAccuracy()
# Applying Random Forest gives me a root mean square deviation = 95.8. It's better than wK Nearest Neighbours.
# However it still worest than Linear Regression and MARS.

#This is how the plot looks like:
randomfModel.randomfPlot()


# %%
#Create an object of XGBoostModel class from Modeling.py file
xgboostModel = XGBoostModel(df)

xgboostModel.xgboostAccuracy()
# Applying XGBoost is the best with root mean square deviation = 31.7.

#This is how the plot looks like:
xgboostModel.xgboostPlot()

# %%
arimaModel = ARIMAModel(newData)

arimaModel.srationaryChick()

arimaModel.ARIMAAccuracy(train, valid)

arimaModel.ARIMAPlot(train, valid)

# %%

lstmModel = LSTMModel(df)

lstmModel.LSTMAccuracy()

lstmModel.LSTMPlot()


# %%
