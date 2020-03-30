#!/usr/bin/env python
# coding: utf-8

# # Submission 1: Basic Statistics, Linear Regression, and Univariate Analysis

# This submission requires you to implement linear regression in forecasting and analysis. Linear regression represents a basic econometric tool and it is the starting point for a variety of prediction models such as ARMA and non-linear algorithms. You will also use the Box-Jenkins approach for selecting the optimal parameters of ARMA forecasting model. Exogenous variables, which improve model forecasts, must be indicated at the end of this submission, to combine with finance theory and research.
# 

# ## Basic Statistics

# Download JP Morgan stock historical prices from Yahoo Finance with the following characteristics:
# 
# - Period: February1,2018 – December30,2018
# - Frequency: Daily
# - Priceconsidered in the analysis: Close price adjusted for dividends and splits

# In[5]:


# importing libraries
import pandas as pd
import pandas_datareader.data as pdr
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pandas.plotting import autocorrelation_plot as ac_plot


# In[6]:


# setting up seaborn
sns.set(rc={
    "figure.figsize": (10,5)
})


# In[7]:


# pdr.DataReader?
jpm = pdr.DataReader("JPM", start="2018-02-01", end="2018-12-30", data_source="yahoo")['Adj Close']
jpm.head()


# In[8]:


jpm.plot()


# **Using this data and R as the programming language, calculate the following:**
# 1. Average stock value
# 2. Stock volatility
# 3. Daily stock return

# In[9]:


# Average Stock value
avgStockValue = jpm.mean()
print("Average Stock Value is $"+str(round(avgStockValue, 3)))


# In[10]:


# Stock Volatility
stockVol = jpm.std()
print("Stock Volatility is "+str(round(stockVol, 3)))


# In[11]:


# Daily Stock Return 
# we will use average compunded daily return formula:  daily return = 1 - [(1+r1)(1+r2).......(1+rn)]^1/n
daily_return = (1- jpm.pct_change().dropna().apply(lambda x: 1+x).cumprod()[-1])*100
print("Average Compounded Daily Stock Return: "+ str(round(daily_return,2))+"%")


# Using the same data above, calculate the following in Excel (you can use OpenOffice as an alternative to Excel):
# 1. Average stock value
# 2. Stock volatility
# 3. Daily stock return
# 4. Show JP Morgan stock price evolution using a scatter plot
# 5. Add a trendline to the graph (trendline options – linear)

# ## Linear Regression

# Implement a two-variable regression in R language using the following data: 
# - Explained variable: JPMorganstock (adjusted close price)
# - Explanatory variable: S&P500
# - Period: February1, 2018–December30, 2018
# - Frequency: Daily

# In[12]:


# get JMP and SP500(^GSPC) data
stocks_df = pdr.DataReader(["JPM", "^GSPC"], start="2018-02-01", end="2018-12-30", data_source="yahoo")['Adj Close']
stocks_df.head()


# In[13]:


stocks_df.plot()


# In[14]:


# set explanatory variable
x = stocks_df['^GSPC']
# adding constant to the OLS
X = sm.add_constant(x)
# set explained variable 
Y = stocks_df['JPM']

# model of the regression
model = sm.OLS(Y,X)

# results
result = model.fit()
print(result.summary())


# **With the same variables as above (JP Morgan Stock and S&P500), implement a two-
# variable regression in Excel using LINEST function and Analysis ToolPak.**
# 

# ## Univariate Time Series

# Download the following data:
# 
# - Datasource: https://fred.stlouisfed.org/series/CSUSHPISA 
# - Period considered in the analysis: January 1987 – latestdata 
# - Frequency: monthly data

# In[15]:


# getting the data from fred data source
csu_data = pdr.DataReader("CSUSHPISA", data_source="fred", start="1987-01-01")
csu_data.head()


# ### With this data, do the following using R or Python languages:

# #### 1. Forecast S&P/Case-Shiller U.S. National Home Price Index using an ARMA model.

# In[16]:


# dir(sm.tsa.stattools)
# sm.tsa.stattools.ARMA?
# dir(sm.tsa)
data_acf = sm.tsa.acf(csu_data, nlags=10)
ac_plot(csu_data.values)


# In[17]:


plt.plot(sm.tsa.pacf(csu_data, nlags=10))


# In[18]:


result = sm.tsa.ARMA(csu_data,(1,0)).fit()


# In[25]:


print(result.summary())
result.plot_predict(start="2015-12-01", end="2020-12-01")


# #### 2. Implement the Augmented Dickey-Fuller Test for checking the existence of a unit root in Case-Shiller Index series.

# In[26]:


# checking for adfuller test
print(sm.tsa.stattools.adfuller(csu_data))


# In[27]:


# taking first difference series
csu_data.diff().plot()


# In[28]:


# check if the first difference series is stationary
print(sm.tsa.stattools.adfuller(csu_data.diff().dropna()))


# Here p-value(second term in the result) is high (p = 0.94190) for first series. which fails to reject the null hypothesis. so the we accept the null hypothesis which is the series is non-stationary
# 
# but If we take the first difference of the series then do the ADF test. The p value becomes .029, which is very low, so we reject the null hypothesis and accept the alternate hypothesis, as the the series is stationary. 
# 

# #### 3. Implement an ARIMA(p,d,q) model. Determine p, d, q using Information Criterion or Box- Jenkins methodology. Comment the results.

# In[29]:


# from above example the difference needed to make the series stationary is 1
stat_data = csu_data.diff().dropna()


# In[30]:


# Box-jenkins method to determin AR/MA order
plt.plot(sm.tsa.stattools.acf(stat_data, nlags=10))


# In[31]:


plt.plot(sm.tsa.stattools.pacf(stat_data, nlags=10))


# Here as we can see, the first difference of the index data is stationary. so d = 1, If we take autocorrelation function and partial autocorrelation of the difference series, then the acf is gradually declining and the pacf is drastically went down after 1 lag. So, first lag explains the most of the relations. Using box-jenkins method. Its a AR series and the order of AR is 1 (from pacf spike). And MA order is 0. so (p,d,q) = (1,1,0)

# In[32]:


# so its a AR series, order of AR is p= 1, MA, q = 0 and d = 1; so ARIMA(1,1,0) model will be best fit
arima_model = sm.tsa.ARIMA(csu_data, order=(1,1,0))
arima_result = arima_model.fit()
print(arima_result.summary())
# .plot_predict(start="2015-12-01", end="2020-12-01")
forecast_df = arima_result.predict(start="2015-12-01", end="2020-12-01")
print(forecast_df.tail(7))
forecast_plot = arima_result.plot_predict(start="2015-12-01", end="2020-12-01")


# #### 4. Forecast the future evolution of Case-Shiller Index using the ARMA model. Test model using in-sample forecasts.

# In[33]:


# forcasting the future evolution
# spliting the data into model and test set
model_data = csu_data[:"2010"]
test_data = csu_data["2011":]
start = test_data.index.values[0]
end = test_data.index.values[-1]

model_train = sm.tsa.ARIMA(model_data, order=(1,1,0)).fit()


# In[34]:


model_forecast= model_train.predict(start, end)
model_train.plot_predict(start="2000-01-01", end="2018-01-01")
csu_data['2000-01-01': '2018-01-01'].plot()


# **Submission Requirements**
# 
# - Required length for your report: about 500 words. Submit your report in a PDF document separate from the rest of the documents.
# - Submit Excel spreadsheets and the source code separately. Add appropriate comments to explain how it works.

# In[ ]:




