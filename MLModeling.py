#%% import libraries:

import pandas as pd
import numpy as np

#to plot within notebook
import matplotlib.pyplot as plt

#for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

#implement modeles 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier

#importing libraries for KNearestNeighbours Model
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV

#I can not push to git hub while useing this package.
from pyearth import Earth


#RMSE
from sklearn.metrics import mean_squared_error

#@staticmethod
def timeToFloat(timestamp):
    toStr = timestamp['Date'].astype('str')
    floats = []
    
    for index, value in toStr.items():
        floats.append([float(x) for x in value.split('-')])

    return floats

# %% Linear Regression
class LinearRegressionModel:

    def __init__(self, df):
        self.df = df

    def linearAccuracy(self, x_train, y_train, x_valid, y_valid):
        #setting index as date values
        self.df.index = self.df['Date']
       
        x_train = timeToFloat(x_train)
        x_valid = timeToFloat(x_valid)

        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(x_train,y_train)

        #Results:
        #make predictions and find the rmse
        self.preds = model.predict(x_valid)
        
        rmse = np.sqrt(mean_squared_error(y_valid, self.preds))
        return rmse



    def linearPlot(self, train, valid):
        #plot
        valid['Predictions'] = 0
        valid['Predictions'] = self.preds

        valid.index = self.df[200:].index
        train.index = self.df[:200].index

        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])

#%%
class KNearestNeighboursModel:

    def __init__(self, df):
        self.df = df

    
    def knnAccuracy(self, x_train, y_train, x_valid, y_valid):
        
        x_train = timeToFloat(x_train)
        x_valid = timeToFloat(x_valid)


        #scaling data
        x_train_scaled = scaler.fit_transform(x_train)
        x_train = pd.DataFrame(x_train_scaled)
        x_valid_scaled = scaler.fit_transform(x_valid)
        x_valid = pd.DataFrame(x_valid_scaled)
    
        
        #using gridsearch to find the best parameter
        params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
        knn = neighbors.KNeighborsRegressor()
        model = GridSearchCV(knn, params, cv=5)

        #fit the model and make predictions
        model.fit(x_train,y_train)
        self.preds = model.predict(x_valid)

        #Result
        #rmse
        rmse = np.sqrt(mean_squared_error(y_valid, self.preds))
        return rmse

    def knnPlot(self, train, valid):
        #plot
        valid['Predictions'] = 0
        valid['Predictions'] = self.preds

        valid.index = self.df[200:].index
        train.index = self.df[:200].index

        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])
        

#%%


class MARSmodel:
    def __init__(self, df):
        self.df = df

    def marsAccuracy(self, x_train, y_train, x_valid, y_valid):

        x_train = timeToFloat(x_train)
        x_valid = timeToFloat(x_valid)
        
        # define the model
        model = Earth()

        # fit the model on training dataset
        model.fit(x_train, y_train)
        self.preds = model.predict(x_valid)

        #Result
        #rmse
        rmse = np.sqrt(mean_squared_error(y_valid, self.preds))
        return rmse


    def marsPlot(self, train, valid):
        #plot
        valid['Predictions'] = 0
        valid['Predictions'] = self.preds

        valid.index = self.df[200:].index
        train.index = self.df[:200].index

        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])

# %%

class  RandomforestModel:

    def __init__(self, df):
        self.df = df

    def randomfAccuracy(self, x_train, y_train, x_valid, y_valid):


        x_train = timeToFloat(x_train)
        x_valid = timeToFloat(x_valid)

        # Instantiate model with 1000 decision trees
        model = RandomForestRegressor(n_estimators = 100, random_state = 32)
        
        
        # Train the model on training data
        model.fit(x_train,y_train)
        
        #Result
        # Use the forest's predict method on the test data
        self.preds = model.predict(x_valid)

        
        #rmse
        rmse = np.sqrt(mean_squared_error(y_valid, self.preds))
        return rmse

    def randomfPlot(self, train, valid):
        #plot
        valid['Predictions'] = 0
        valid['Predictions'] = self.preds

        valid.index = self.df[200:].index
        train.index = self.df[:200].index

        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])


# %%
from xgboost import XGBRegressor 

class XGBoostModel:
    def __init__(self, df):
        self.df = df

    
    def xgboostAccuracy(self, x_train, y_train, x_valid, y_valid):


        x_train = timeToFloat(x_train)
        x_valid = timeToFloat(x_valid)

        # converting list to array
        X_train = np.array(x_train)
        X_valid = np.array(x_valid)
        
        xgb = XGBRegressor()
        xg_reg = xgb.fit(X_train, y_train)
        
        
        self.preds = xg_reg.predict(X_valid)

        rmse = np.sqrt(mean_squared_error(y_valid, self.preds))
        return rmse

    
    def xgboostPlot(self, train, valid):
        #plot
        valid['Predictions'] = 0
        valid['Predictions'] = self.preds

        valid.index = self.df[200:].index
        train.index = self.df[:200].index

        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])