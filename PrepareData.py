
#%%import packages
import pandas as pd
import numpy as np

#to plot within notebook
import matplotlib.pyplot as plt

#for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

#Removing features with low variance using scikit-learn
from sklearn.feature_selection import VarianceThreshold


#@staticmethod
def timeToFloat(timestamp):
    toStr = timestamp['Date'].astype('str')
    floats = []
    
    for index, value in toStr.items():
        floats.append([float(x) for x in value.split('-')])

    return floats

#%%
class PreparingData:

    def __init__(self, df):
        self.df = df
    
    
    # setter method 
    def setDf(self, df):
        self.df = df

    # getter method 
    def getDf(self):
        return self.df

   
    def fillNa(self, y_train):
        #fill missing values with mean column values
        #It'll not give us the right result if we apply it to date
        return y_train.fillna(y_train.mean(), inplace=True)

    
    def RmLowVar(self):

        #Removing features with low variance using scikit-learn.
        #parameter threshold in which we have to put the minimum value of variance we want in out dataset.
        #Then we have used fit_transform to fit and transform the dataset.
        import pandas as pd 
        import numpy 
       
        thresholder = VarianceThreshold()
        #y_HighVariance = thresholder.fit_transform(numpy.reshape(y_valid ,-1))
        
        self.df = timeToFloat(self.df)
        y_HighVariance = thresholder.fit_transform(self.df)

        return y_HighVariance[0:5]

        #plot high variance
        #plt.plot(y_HighVariance)

    
    def Correlation(self):
    #https://stackabuse.com/applying-filter-methods-in-python-for-feature-selection/    
        # To find the correlation among 
        # the columns using kendall method 

        correlatedfeatures = self.df.corr()
        #correlatedfeatures = y_train.corr(method ='pearson', min_periods=1) 

        #correlated_features = set()
        #correlation_matrix = x_train.corr()

        #for i in range(len(correlation_matrix.columns)):
        #    for j in range(i):
        #        if abs(correlation_matrix.iloc[i, j]) > 0.8:
        #            colname = correlation_matrix.columns[i]
        #            correlated_features.add(colname)

        return correlatedfeatures
