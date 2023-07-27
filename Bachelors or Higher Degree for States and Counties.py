#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('Bachelors or Higher Degree for States and Counties.csv')


# In[3]:


df.head()


# In[4]:


df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'])

df_filled = df.fillna(0)

df_filled = df_filled.loc[:, (df_filled != 0).any(axis=0)]

df_filled = df_filled.loc[(df_filled != 0).any(axis=1)]

df_filled.head()


# In[5]:


def simplify_name(name):
    if "Bachelor's Degree or Higher for " in name:
        return name.replace("Bachelor's Degree or Higher for ", "")
    if "Bachelor's Degree or Higher (5-year estimate) in " in name:
        return name.replace("Bachelor's Degree or Higher (5-year estimate) in ", "")
    return name

df_filled.columns = df_filled.columns.map(simplify_name)

df_filled.head()


# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans


# In[7]:


sns.set(style="whitegrid")
random_states = np.random.choice(df_filled.columns[1:], size=5)


# In[8]:


correlation_matrix = df_filled[random_states].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix for Selected States')
plt.show()


# In[10]:


states = ['New York County, NY', 'Letcher County, KY', 'Evans County, GA', 'Bristol County, MA', 'Harney County, OR']

plt.figure(figsize=(14, 8))
for state in states:
    plt.plot(df_filled['Unnamed: 0'], df_filled[state], label=state)
plt.legend()
plt.xlabel('Year')
plt.ylabel('Percentage with Bachelor\'s Degree or Higher')
plt.title('Trend of Percentage with Bachelor\'s Degree or Higher Over Time')
plt.show()


# In[11]:


from statsmodels.tsa.arima.model import ARIMA
from pandas.tseries.offsets import DateOffset


# In[12]:


def make_forecast(state):
    model = ARIMA(df_filled[state], order=(1,1,1))
    model_fit = model.fit()

    future_dates = [df_filled['Unnamed: 0'].max() + DateOffset(years=x) for x in range(0,6)]
    forecast = model_fit.predict(start=len(df_filled), end=len(df_filled)+4)

    plt.figure(figsize=(10,6))
    plt.plot(df_filled['Unnamed: 0'], df_filled[state], label='Original')
    plt.plot(future_dates[1:], forecast, label='Forecast')
    plt.title(f'Forecast for {state}')
    plt.xlabel('Year')
    plt.ylabel('Percentage with Bachelor\'s Degree or Higher')
    plt.legend()
    plt.show()

for state in states:
    make_forecast(state)


# In[ ]:




