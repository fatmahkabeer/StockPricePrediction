# StockPricePrediction

I have got a dataset from https://www.quandl.com to implement a Machain Learning to predict a stock price. 

### This repository contains five files:
- *StockPricePrediction.py* is the main file, where all other files are imported to executed.
- I implemented some function to discover a dataset in a file called *DiscoveringDate.py*
- *PrepareData.py*, has functions to prepare data for modeling.
- *MLModeling.py* has six classes for different six models:
     - 1. LinearRegressionModel
     - 2. KNearestNeighboursModel
     - 3. MARSmodel
     - 4. RandomforestModel
     - 5. XGBoostModel
     - 6. ARIMA
- *DLModeling.py*, has LSTM class. 


NOTE:
Version 3.6.8 of python has been used. 
