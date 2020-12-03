# I wanted to start working in this part, 
#however, because of a deadline and having to start working on Capstone project I delay that to after graduate.


#%% import libraries:

import pandas as pd
import numpy as np

#to plot 
import matplotlib.pyplot as plt

#to implement ARIMA
from statsmodels.tsa.arima_model import ARIMA

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


training = train['Close']
validation = valid['Close']

#%%# fit model
model = ARIMA(training, order=(1,1,1))
fitted = model.fit(disp=0)