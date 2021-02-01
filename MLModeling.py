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


#to implement ARIMA
from pmdarima.arima import auto_arima

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

    def linearAccuracy(self):
        #setting index as date values
        self.df.index = self.df['Date']

        self.train = self.df[:200]
        self.valid = self.df[200:]
       
        #Split data:
        x_train = self.train.drop('Close', axis=1)
        y_train = self.train['Close']

        x_valid = self.valid.drop('Close', axis=1)
        y_valid = self.valid['Close']

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



    def linearPlot(self):
        #plot
        self.valid['Predictions'] = 0
        self.valid['Predictions'] = self.preds

        self.valid.index = self.df[200:].index
        self.train.index = self.df[:200].index

        plt.plot(self.train['Close'])
        plt.plot(self.valid[['Close', 'Predictions']])
        plt.title('Linear Regression')
        plt.show()

#%%
class KNearestNeighboursModel:

    def __init__(self, df):
        self.df = df

    
    def knnAccuracy(self):

        #setting index as date values
        self.df.index = self.df['Date']

        self.train = self.df[:200]
        self.valid = self.df[200:]
       
        #Split data:
        x_train = self.train.drop('Close', axis=1)
        y_train = self.train['Close']

        x_valid = self.valid.drop('Close', axis=1)
        y_valid = self.valid['Close']

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

    def knnPlot(self):
        #plot
        self.valid['Predictions'] = 0
        self.valid['Predictions'] = self.preds

        self.valid.index = self.df[200:].index
        self.train.index = self.df[:200].index

        plt.plot(self.train['Close'])
        plt.plot(self.valid[['Close', 'Predictions']])
        plt.title('KNN')
        plt.show()

#%%


class MARSmodel:
    def __init__(self, df):
        self.df = df

    def marsAccuracy(self):

        #setting index as date values
        self.df.index = self.df['Date']

        self.train = self.df[:200]
        self.valid = self.df[200:]
       
        #Split data:
        x_train = self.train.drop('Close', axis=1)
        y_train = self.train['Close']

        x_valid = self.valid.drop('Close', axis=1)
        y_valid = self.valid['Close']

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


    def marsPlot(self):
        #plot
        self.valid['Predictions'] = 0
        self.valid['Predictions'] = self.preds

        self.valid.index = self.df[200:].index
        self.train.index = self.df[:200].index

        plt.plot(self.train['Close'])
        plt.plot(self.valid[['Close', 'Predictions']])
        plt.title('MARS')
        plt.show()

# %%

class  RandomforestModel:

    def __init__(self, df):
        self.df = df

    def randomfAccuracy(self):


        #setting index as date values
        self.df.index = self.df['Date']

        self.train = self.df[:200]
        self.valid = self.df[200:]
       
        #Split data:
        x_train = self.train.drop('Close', axis=1)
        y_train = self.train['Close']

        x_valid = self.valid.drop('Close', axis=1)
        y_valid = self.valid['Close']

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

    def randomfPlot(self):
        #plot
        self.valid['Predictions'] = 0
        self.valid['Predictions'] = self.preds

        self.valid.index = self.df[200:].index
        self.train.index = self.df[:200].index

        plt.plot(self.train['Close'])
        plt.plot(self.valid[['Close', 'Predictions']])
        plt.title('Random Forest')
        plt.show()
        


# %%
from xgboost import XGBRegressor 

class XGBoostModel:
    def __init__(self, df):
        self.df = df

    
    def xgboostAccuracy(self):


        #setting index as date values
        self.df.index = self.df['Date']

        self.train = self.df[:200]
        self.valid = self.df[200:]
       
        #Split data:
        x_train = self.train.drop('Close', axis=1)
        y_train = self.train['Close']

        x_valid = self.valid.drop('Close', axis=1)
        y_valid = self.valid['Close']

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

    
    def xgboostPlot(self):
        #plot
        self.valid['Predictions'] = 0
        self.valid['Predictions'] = self.preds

        self.valid.index = self.df[200:].index
        self.train.index = self.df[:200].index

        plt.plot(self.train['Close'])
        plt.plot(self.valid[['Close', 'Predictions']])
        plt.title('XGBoost')
        plt.show()




#%%

class ARIMAModel:
#use the ‘Augmented Dickey-Fuller Test’ to check whether the data is stationary or not. 

    def __init__(self, df):
        self.df = df
    
    
    def srationaryChick(self):
        from pmdarima.arima import ADFTest

        adfTest = ADFTest(alpha = 0.05)
        print(adfTest.should_diff(self.df))


    def ARIMAAccuracy(self, train, valid):

        training = train['Close']
        validation = valid['Close']

        #In the Auto ARIMA model, note that small p,d,q values represent non-seasonal components, and capital P, D, Q represent seasonal components.
        #It works similarly like hyper tuning techniques to find the optimal value of p, d, and q with different combinations and the final values would be determined with the lower AIC, BIC parameters taking into consideration.
        model = auto_arima(training, start_p = 0, start_q = 0, max_p = 5, max_q = 5, m = 12,start_P = 0, seasonal = True, d = 1, D = 1, trace = True, error_action='warn', suppress_warnings = True)
        model.fit(training)

        self.forecast = model.predict(n_periods = 42)
        self.forecast = pd.DataFrame(self.forecast,index = valid.index,columns=['Prediction'])

        rms = np.sqrt(np.mean(np.power((np.array(valid['Close'])-np.array(self.forecast['Prediction'])),2)))

        return rms


    def ARIMAPlot(self, train, valid):
        #plot
        plt.plot(train['Close'])
        plt.plot(valid['Close'])
        plt.plot(self.forecast['Prediction'])
        plt.title('ARIMA')
        plt.show()
# %%

