# I wanted to start working on this part,
# however, because of a deadline and have to start working on the Capstone project I delay that until after graduation.

#%% import libraries:

import pandas as pd
import numpy as np

#to plot 
import matplotlib.pyplot as plt

#%%
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#%%
#converting dataset into x_train and y_train
#scaler = MinMaxScaler(feature_range=(0, 1))
#scaled_data = scaler.fit_transform(dataset)


#%%
#LSTMs expect our data to be in a specific format, usually a 3D array.
#We start by creating data in 60 timesteps and converting it into an array using NumPy.
#Next, we convert the data into a 3D dimension array with X_train samples,
#60 timestamps, and one feature at each step.


class LSTMModel:

    def __init__(self, df):
        self.df = df


    def LSTMAccuracy(self):
        #creating dataframe
        data = self.df.sort_index(ascending=True, axis=0)
        self.newData = pd.DataFrame(index=range(0,len(self.df)),columns=['Date', 'Close'])
        
        for i in range(0,len(data)):
            self.newData['Date'][i] = data['Date'][i]
            self.newData['Close'][i] = data['Close'][i]

        #setting index
        self.newData.index = self.newData.Date
        self.newData.drop('Date', axis=1, inplace=True)


        #creating train and test sets
        dataset = self.newData.values


        train = dataset[0:200,:]
        valid = dataset[200:,:]

        #converting dataset into x_train and y_train
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)


        x_train, y_train = [], []
        for i in range(60,len(train)):
            x_train.append(scaled_data[i-60:i,0])
            y_train.append(scaled_data[i,0])
        x_train, y_train = np.array(x_train), np.array(y_train)

        x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


        # create and fit the LSTM network
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))

        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

        #predicting 246 values, using past 60 from the train data
        inputs = self.newData[len(self.newData) - len(valid) - 60:].values
        inputs = inputs.reshape(-1,1)
        inputs  = scaler.transform(inputs)

        X_test = []
        for i in range(60,inputs.shape[0]):
            X_test.append(inputs[i-60:i,0])
        X_test = np.array(X_test)

        X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
        self.closing_price = model.predict(X_test)
        self.closing_price = scaler.inverse_transform(self.closing_price)

        rms=np.sqrt(np.mean(np.power((valid - self.closing_price),2)))
   
        return rms


    def LSTMPlot(self):
        #for plotting
        train = self.newData[:200]
        valid = self.newData[200:]
        valid['Predictions'] = self.closing_price
        plt.plot(train['Close'])
        plt.plot(valid[['Close','Predictions']])
        plt.title('LSTM')
        plt.show()

