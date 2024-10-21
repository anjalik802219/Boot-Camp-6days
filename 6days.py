#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt 


# In[2]:


df = pd.read_csv("C:\\Users\Anjali Kumari\\OneDrive\\Desktop\\archive\\car data.csv")


# In[3]:


df.head


# In[4]:


df.columns


# In[6]:


df.describe()


# In[7]:


df1 = pd.read_csv("C:\\Users\Anjali Kumari\\OneDrive\\Desktop\\archive\\CAR DETAILS FROM CAR DEKHO.csv")


# In[8]:


df2 = pd.read_csv("C:\\Users\Anjali Kumari\\OneDrive\\Desktop\\archive\\Car details v3.csv")


# In[9]:


df3 = pd.read_csv("C:\\Users\Anjali Kumari\\OneDrive\\Desktop\\archive\\car details v4.csv")


# In[10]:


df1.head


# In[11]:


df2.head


# In[12]:


df3.head()


# In[13]:


df1.info


# In[14]:


df.info()


# In[15]:


df2.info()


# In[16]:


df.isnull().sum()


# In[17]:


df1.isnull().sum()


# In[18]:


df2.isnull().sum()


# In[19]:


df3.isnull().sum()


# In[20]:


corr = df1.corr()
corr


# In[21]:


corr = df2.corr()
corr


# In[23]:


corr = df3.corr()
corr


# In[24]:


sns.pairplot(df1)


# In[27]:


sns.pairplot(df2)


# In[28]:


sns.pairplot(df3)


# In[34]:


import pandas as pd

# Assuming df1 is already defined

# Print the columns in df1 to debug
print("Columns in df1:", df1.columns)

# Define a list of columns to drop
columns_to_drop = ['Price', 'Year', 'Kilometer', 'Length', 'Width']

# Filter out columns that are not in the DataFrame
existing_columns_to_drop = [col for col in columns_to_drop if col in df1.columns]

# Drop only the existing columns
df1.drop(existing_columns_to_drop, axis=1, inplace=True)

# Display the first few rows of the modified DataFrame
print(df1.head())


# In[37]:


sns.distplot(df1['year'])
plt.show()


# In[38]:


import pandas as pd

# Assuming df1 is already defined

# Print the columns in df1 to debug
print("Columns in df2:", df2.columns)

# Define a list of columns to drop
columns_to_drop = ['Price', 'Year', 'Kilometer', 'Length', 'Width']

# Filter out columns that are not in the DataFrame
existing_columns_to_drop = [col for col in columns_to_drop if col in df2.columns]

# Drop only the existing columns
df2.drop(existing_columns_to_drop, axis=1, inplace=True)

# Display the first few rows of the modified DataFrame
print(df2.head())


# In[39]:


sns.distplot(df1['year'])
plt.show()


# In[40]:


import pandas as pd

# Assuming df1 is already defined

# Print the columns in df1 to debug
print("Columns in df3:", df3.columns)

# Define a list of columns to drop
columns_to_drop = ['Price', 'Year', 'Kilometer', 'Length', 'Width']

# Filter out columns that are not in the DataFrame
existing_columns_to_drop = [col for col in columns_to_drop if col in df3.columns]

# Drop only the existing columns
df3.drop(existing_columns_to_drop, axis=1, inplace=True)

# Display the first few rows of the modified DataFrame
print(df3.head())


# In[47]:


sns.distplot(df1['year'])
plt.show()


# In[42]:


import pandas as pd

# Assuming df1 is already defined

# Print the columns in df1 to debug
print("Columns in df:", df.columns)

# Define a list of columns to drop
columns_to_drop = ['Price', 'Year', 'Kilometer', 'Length', 'Width']

# Filter out columns that are not in the DataFrame
existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]

# Drop only the existing columns
df.drop(existing_columns_to_drop, axis=1, inplace=True)

# Display the first few rows of the modified DataFrame
print(df.head())


# In[49]:


sns.distplot(df1['year'])
plt.show()


# In[50]:


df.shape


# In[51]:


df1.shape


# In[52]:


df2.shape


# In[53]:


df3.shape


# In[54]:


df.describe().T


# In[55]:


df1.describe().T


# In[56]:


df2.describe().T


# In[57]:


df3.describe().T


# In[58]:


df.dtypes


# In[59]:


df1.dtypes


# In[60]:


df2.dtypes


# In[61]:


df3.dtypes


# In[62]:


df.isnull().sum()


# In[63]:


df1.isnull().sum()


# In[64]:


df2.isnull().sum()


# In[65]:


df3.isnull().sum()


# In[66]:


df.duplicated().sum()


# In[67]:


df1.duplicated().sum()


# In[68]:


df2.duplicated().sum()


# In[69]:


df3.duplicated().sum()


# In[70]:


df[df.duplicated()]


# In[71]:


df1[df1.duplicated()]


# In[72]:


df2[df2.duplicated()]


# In[73]:


df3[df3.duplicated()]


# In[79]:


def PercentageofMissingData(dataset):
    return dataset.isna().sum()/len(dataset*100)
print(df.isnull().sum())
print()
print(PercentageofMissingData(df))


# In[80]:


def PercentageofMissingData(dataset):
    return dataset.isna().sum()/len(dataset*100)
print(df1.isnull().sum())
print()
print(PercentageofMissingData(df1))


# In[81]:


def PercentageofMissingData(dataset):
    return dataset.isna().sum()/len(dataset*100)
print(df2.isnull().sum())
print()
print(PercentageofMissingData(df2))


# In[82]:


df.tail()


# In[83]:


df1.tail()


# In[84]:


df2.tail()


# In[85]:


df.tail()


# In[88]:


print(df1['selling_price'].quantile(0.01))
print(df1['selling_price'].quantile(0.99))


# In[89]:


print(df2['selling_price'].quantile(0.01))
print(df2['selling_price'].quantile(0.99))


# In[91]:


sns.boxplot(df1['selling_price'])
plt.xlabel('Selling_Price')sns.boxplot(raw_data['selling_price'])
plt.xlabel('Selling_Price')


# In[92]:


sns.boxplot(df2['selling_price'])
plt.xlabel('Selling_Price')


# In[94]:


sns.boxplot(df1['selling_price'])
plt.xlabel('Selling_Price')


# In[ ]:




