#!/usr/bin/env python
# coding: utf-8

# # Submission 2: Volatility and Multivariate Analysis

# Financial variables have a time varying mean and periods of relatively low tranquility followed by periods of high variability. For this submission you will extend ARMA Box-Jenkins methodology to stochastic volatility models (GARCH), provide short term forecasts, and identify data patterns. You will also implement cointegration and VECM framework to calculate equilibrium levels for financial variables.
# For this submission you can use either R or Python languages. 
# ## Volatility Analysis
# ### Forecast Apple daily stock return using a GARCH model. Use Yahoo Finance as your data source.
# 1. Select one type of GARCH model (ARCH, GARCH-M, IGARCH, EGARCH, TARCH,multivariate GARCH, etc.) to complete your analysis. Explain your choice.
# 2. Forecast the next period daily return (t+1) using the chosen model. Select the timeframe in the analysis. Provide charts and comments.
# 
# ## Multivariate Analysis
# ### Calculate the equilibrium FX for your local currency and do the following:
# 1. Describe the economic theories and models used to calculate equilibrium FX.
# 2. Indicate macroeconomic variables used to determine the equilibrium FX.
# 3. Explain the connection between linear regression and Vector Error Correction (VEC).
# 4. Calculate the equilibrium FX using VEC and comment all your results. You may use the Behavioral Equilibrium Exchange Rate (BEER) approach
# 
# **In the written report for this submission, provide four (4) research articles or books at minimum. Submission Requirements**
# 
# **Required length for your report: 2 pages or about 1,000 words.** Submit your report in a PDF document.
# Submit your source code separately. Add appropriate comments to explain how it works.

# # Volatility Analysis

# In[1]:


import warnings; warnings.simplefilter('ignore')


# In[2]:


import pandas as pd
import numpy as np
import statsmodels.tsa.api as smt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from arch import arch_model
import pandas_datareader.data as pdr
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(rc={"figure.figsize": (10,7)})
import missingno


# ### Getting Apple Stock Data 

# Data taken from yahoo finance from January 2018 to December 2019. We have taken the log return of the AAPL Adjusted Closing price. It is looking like a stationary series with variable volatility. Lets investigate more. 

# In[3]:


data = pdr.DataReader("AAPL", data_source="yahoo", start="2018-01-01", end = "2019-12-30" )['Adj Close']
daily_return = np.log(data/data.shift(1)).dropna()
daily_return.plot()


# ### Create a standard Timeseries Plot for analysis
# For common analysis we will be needing a standard time series plot with ACF and PACF plot to carry out necessary analysis. Lets create a function that plots those. Then we will plot the daily return values in the time series plot. 
# 
# At 30 lags both ACF and PACF plot shows similar characteristics. We cant be sure as it matches AR models or MA models. 

# In[4]:


def tsplot(y, lags=None):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    layout = (2,2)
    ts_ax = plt.subplot2grid(layout, (0,0), colspan=2 )
    acf_ax = plt.subplot2grid(layout, (1,0))
    pacf_ax = plt.subplot2grid(layout, (1,1) )
    
    y.plot(ax=ts_ax)
    ts_ax.set_title("Time Series Analysis Plot")
    smt.graphics.plot_acf(y,lags=lags, ax=acf_ax, alpha=.05)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=.05)
    
    plt.tight_layout()   


# In[5]:


tsplot(daily_return,lags=30)


# It is clear that the returns have significant correlation with the lags. Lets figure out which Autoregressive Model will be best fit for the time series. 

# ### ARIMA Model
# we will fit the best possible model by checking various orders and get the model with lowest AIC value. 

# In[6]:


def best_fit_model(ts):
    best_aic = np.inf
    best_order = None
    best_model = None
    
    pq_range = range(7)
    d_range = range(2)
    for p in pq_range:
        for d in d_range:
            for q in pq_range:
                try:
                    temp_model = smt.ARIMA(ts,order=(p,d,q)).fit(method="mle", trend="nc")
                    temp_aic = temp_model.aic
                    if temp_aic < best_aic:
                        best_aic = temp_aic
                        best_order = (p,d,q)
                        best_model = temp_model
                except: continue
    print("Best AIC: {:6.5f} | Best Order: {}".format(best_aic,best_order))
    return best_model, best_order, best_aic


# In[7]:


best_result = best_fit_model(daily_return)


# In[55]:


model = best_result[0]
order = best_result[1]


# In[56]:


model.summary()


# All of the lag terms in model shows significant p values. Lets try to plot the residue errors. It is clear that residue has some autocorrelation left. Which becomes significant if we plot the square of the residue. 

# In[57]:


tsplot(model.resid, lags=30)


# In[58]:


tsplot(model.resid**2, lags=30)


# ### GARCH Model 
# Lets consider a GARCH model for the variable volatility in the residue. Which causing the autocorrelation in the error terms. We would take the order from the ARIMA Model and apply it to the arch_model function to fit in the ARCH. Here R squared value is very low. so the model is not fitted well. 

# In[91]:


p_ = order[0]
o_ = order[1]
q_ = order[2]

model_garch = arch_model(daily_return, vol="GARCH", p = p_, o=o_, q=q_,dist="Normal" )


# In[92]:


garch_res = model_garch.fit(update_freq=5, disp='off')
garch_res.summary()


# In[66]:


tsplot(garch_res.resid, lags=30)


# In[67]:


tsplot(garch_res.resid**2, lags=30)


# ### Lets Forecast 
# we will forcast one step into the future
# 

# In[77]:


forecast_arch = garch_res.forecast(horizon=5)


# In[78]:


forecast_arch.mean.iloc[-3:]


# # Multivariate Analysis

# ### Equilibrium FX 
# For equilibrium fx we have taken bdt as local currency and Usd as forein currency. CPI in US, Imports and Exports data from the US to Bangladesh as independent variables. We have taken data from the FRED. There were some seasonality in the data, we removed the seasonlity and adjusted the dataset. 

# In[18]:


symbols_fred = ["CPIAUCSL", "EXP5380", "IMP5380"]
symbols_yahoo = ["USDBDT=X"]


# In[19]:


data_us = pdr.DataReader(symbols_fred, data_source="fred", start="2003-01-10", end="2020-01-01").resample("M").mean()
data_bd = pdr.DataReader(symbols_yahoo, data_source="yahoo", start="2003-01-10", end="2020-01-01")['Close'].resample("m").mean()
data_bd = data_bd[1:]
data = data_us.join(data_bd)
for each in data.columns:
    res = seasonal_decompose(data[each], model="additive")
    data[each+'adj'] = data[each] - res.seasonal
data.head()


# In[20]:


fig, ax = plt.subplots(nrows=4, ncols=2)
for i in range(len(data.columns)):
    data.iloc[:,i].plot(ax=ax.flatten()[i])
    ax.flatten()[i].set_title(data.columns[i])    
plt.tight_layout()


# In[21]:


adj_data = data.iloc[:,4:]


# In[22]:


adj_data.plot(subplots=True)


# ### Grangers Causality Test
# We have done grangers causality test to figure out the which variable lags impacts the other variables. Clearly Export and import data are independent. But CPI and FX is dependent to some extent to other variables. 

# In[23]:


gc_matrix = pd.DataFrame(np.zeros([len(adj_data.columns),len(adj_data.columns)]), columns=adj_data.columns, index=adj_data.columns)


# In[24]:


for y in gc_matrix.columns:
    for x in gc_matrix.index:
        res = grangercausalitytests(adj_data[[y,x]], maxlag=6, verbose=False)
        p_values = [round(res[i+1][0]["ssr_chi2test"][1], 4) for i in range(6)]
        min_values = np.min(p_values)
        gc_matrix.loc[x,y] = min_values


# In[25]:


gc_matrix


# ### Cointegraton Test
# Lets see if the variables we considered are cointegrated. For that we have used Johansen test. The result shows no cointegration for the four variables. But If we take first difference of the series, It becomes cointegrated. 

# In[26]:


res = coint_johansen(adj_data, -1, 5)
for i in range(len(adj_data.columns)):
    print((res.lr1[i]>res.cvt[:,1][i]))


# Here for each series, trace statistics is greater than critical value so, the series is cointegrated. At 95% confidence intarval.

# In[27]:


res = coint_johansen(adj_data.diff(1).dropna(), -1, 5)
for i in range(len(adj_data.columns)):
    print((res.lr1[i]>res.cvt[:,1][i]))


# In[28]:


len(adj_data)


# In[29]:


adj_data.head()


# Lets divide the data into training and testing set

# In[30]:


df_train, df_test = adj_data[:-10], adj_data[-10:]


# In[31]:


df_train.shape


# In[32]:


df_test.shape


# We will check if the simple series is stationary with Augmented Dickey Fuller test. The simple series is not stationary. But for the first difference. all of the series becomes stationary. 

# In[33]:


for each in df_train.columns:
    res = adfuller(df_train[each])
    if res[1] < .05:
        print("Series "+each+" is Stationary")
    else:
        print("Series "+each+" is Non-Stationary")


# In[34]:


for each in df_train.columns:
    res = adfuller(df_train[each].diff(1).dropna())
    if res[1] < .05:
        print("Series "+each+" is Stationary")
    else:
        print("Series "+each+" is Non-Stationary")


# ### VAR model 
# Now we will fit a VAR model for the variables. To get the order of the var model we have loop through various order then chose the model with lowest AIC value. Here for order 2 we have gotten the lowest AIC. So, we have fitted the VAR with order 2. 

# In[35]:


df_tdiff = df_train.diff(1).dropna()
var_model = smt.VAR(df_tdiff)


# In[36]:


best_aic = np.inf
order = None
fitted_model = None
for i in range(7):
    res = var_model.fit(i+1)
    print(res.aic)
    if res.aic<best_aic:
        best_aic = res.aic
        order = i+1
        fitted_model = res
    else:
        continue
    


# In[37]:


order


# In[38]:


fitted_model.summary()


# ### Durbin Watson Test 
# We have done durbin watson test to check if the residuals has enough correlation. But the correlation in the residuals seems okay. 

# In[39]:


out = durbin_watson(fitted_model.resid)


# In[40]:


out


# The correlation seems alright

# ### Forecasting VAR
# We have splited the data set into 10 testing set. Now we will forecast the VAR model 10 step and see if it matches our result. 

# In[41]:


lag_order = fitted_model.k_ar
forecast_input = df_tdiff.values[-2:]
forecast_input


# In[42]:


forecast = fitted_model.forecast(y=forecast_input, steps=10)


# In[43]:


df_forecast = pd.DataFrame(forecast, index=adj_data.index[-10:], columns=adj_data.columns)


# In[44]:


df_forecast


# In[45]:


df_start = df_train.iloc[-1:]


# In[46]:


forecasted_data = df_start.append(df_forecast).cumsum()
forecasted_data


# In[96]:


# We now plotted the graph so see the real value from the testing set and our forecasted value. The results are satisfactory. 
forecasted_data['USDBDT=Xadj'].plot()
df_test['USDBDT=Xadj'].plot()


# In[ ]:




