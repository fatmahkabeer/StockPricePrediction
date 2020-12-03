
import pandas as pd

#to plot within notebook
import matplotlib.pyplot as plt
import seaborn as sns 



class DiscData:

      def __init__(self, df):
         self.df = df

      def plotRelation(self):
         sns.pairplot(self.df, diag_kind = 'kde')

     
      def isNan(self):
         return self.df.isna().sum()

    
      def dataShape(self):
         print('The shape of our data is:')
         return self.df.shape

      def firstObs(self):
         #print the head
         return self.df.head()


      def discoverData(self):
         return (self.df.describe(),  self.df.info(), self.df.dtypes)


      def skewed(self):
         #Is the response skewed?
         #skewness = 0 : normally distributed.
         #skewness > 0 : more weight in the left tail of the distribution.
         #skewness < 0 : more weight in the right tail of the distribution. 

         # skip the na values 
         # find skewness in each row 
         skew = self.df.skew(skipna = True)
         print(skew)


      def feVariances(self):
         for i in self.df.loc[:, self.df.columns != 'Date']:
            if(self.df[i].std() == 'nan'):
               print(i, 'zero variance')
            
            print(i, self.df[i].std())



      def plotDataHist(self):
        #setting index as date
         self.df['Date'] = pd.to_datetime(self.df.Date,format='%Y-%m-%d')
         self.df.index = self.df['Date']

        #plot
         #plt.plot(self.df['Open'], self.df['Close'], self.df['Volume'],
         #         self.df['Turnover'],self.df['Pre Settle'], self.df['Pre Settle'],
         #         self.df['Pre Settle'])
         self.df.plot(subplots=True, layout=(2,5))
        

      def plotCloseHist(self):
         self.df['Date'] = pd.to_datetime(self.df.Date,format='%Y-%m-%d')
         self.df.index = self.df['Date']

         #plot
         plt.plot(self.df['Close'], label='Close Price history')