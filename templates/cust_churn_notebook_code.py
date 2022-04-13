#!/usr/bin/env python
# coding: utf-8

# # FINAL YEAR PROJECT - CUSTOMER CHURN PREDICTION
# 

# ## NAME: SACHI SHAH
# ### CLASS: TYIT
# ### ROLL NO: 522

# # 

# ### Importing Libraries

# In[1]:


import pandas as pd #data manipulation operations
import matplotlib.pyplot as plt # for plotting graphs
import seaborn as sns # for plotting graphs
import numpy as np 
from sklearn 
import metrics # for predictive analytics
from sklearn.tree 
import DecisionTreeRegressor
from sklearn.model_selection 
import train_test_split
from sklearn.metrics 
import r2_score,mean_squared_error # accuracy


# ### Importing dataset

# In[2]:


df = pd.read_csv('Churn_Modelling.csv')


# ### Data Preprocessing

# In[3]:


# First 5 rows of the dataset
df.head() 


# In[4]:


# Number of rows and columns in the dataset
df.shape


# In[5]:


# dataypes of the columns
df.dtypes


# In[6]:


# Check columns list and missing values
df.isnull().sum()


# In[7]:


# Get unique count for each variable
df.nunique()


# ### Analyzing the data

# In[8]:


# Pie chart depicting the percentage of churned and retained customers in the dataset

labels = 
'Exited', 'Retained'
sizes = 
[df.Exited[df['Exited']==1].count(), df.Exited[df['Exited']==0].count()]
explode = (0, 0.1) # only "explode" the 2nd slice
colors = ['#ff9999','#66b3ff']
fig1, ax1 = 
plt.subplots(figsize=(10, 8))
ax1.pie(sizes,explode=explode, labels=labels, autopct='%1.1f%%', colors=colors) #autopct display the percent value
plt.title("Percentage of Churned and Retained Customers")
plt.show()


# In[9]:


# categorical plot for depicting the churners based on geography
sns.catplot(y="Geography", 
hue="Exited", kind="count",data=df) #hue is used to encode the points with different colors


# In[10]:


# categorical plot for depicting the churners based on gender
sns.catplot(y="Gender", 
hue="Exited", kind="count", data=df)


# In[11]:


# categorical plot for depicting the reason of leaving of the exited customers based on tenure
df_new = df[df['Exited'] == 1]
sns.catplot(y="Reason for exiting company", hue="Tenure", 
kind="count", aspect=3.3, data=df_new) # aspect gives the Orientation of the plot


# In[12]:


# creating a copy of the main dataframe
data_group=df


# In[13]:


# showing correlation coefficients between variables

corrMatrix = 
data_group[['CreditScore','Age','Tenure','Balance',
'NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Exited']].corr()
fig, ax = 
plt.subplots(figsize=(10,10))  # Sample figsize in inches

# Visualize the correlation matrix
plt.title('Correlation Matrix of Our Numerical Features', pad=20, fontweight='bold') # pad gives the offset of the title from the top of the axes
sns.heatmap(corrMatrix, cmap='Blues', annot=True)


# ## Model 1: Prediction whether the customer will exit or not

# ### Data preparation

# In[14]:


x = 
data_group[['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']]
y = data_group['Exited']


# ### Splitting the dataset

# In[15]:


x_train, x_test, y_train, y_test = 
train_test_split(x, y, test_size=0.3, shuffle=False, random_state=0)
# shuffle false: shuffles the data before splitting
# random_state: return same results for each execution


# ### Implementation of the model

# In[16]:


DecisionTreeRegModel1 = DecisionTreeRegressor()
DecisionTreeRegModel1.fit(x_train,y_train)
y_pred = DecisionTreeRegModel1.predict(x_test)


# In[17]:


print("The predicted values of exited are ", y_pred)


# In[18]:


print("The actual values of exited are ", y_test)


# ### Evaluating the Performance

# In[19]:


# r2_score: used to evaluate the performance of a regression-based machine learning model
r2_score(y_test, y_pred)


# ### Accuracy of the model

# In[20]:


from sklearn.metrics 
import accuracy_score
print(accuracy_score(y_pred, y_test)*100)


# # 

# ## Model 2: Predicting the customers' reason of leaving

# ### Dealing with categorical values

# In[21]:


# Label encoding: each label is converted into an integer value
reason = 
{'High Service Charges/Rate of Interest':0, 'Long Response Times':1, 'Inexperienced Staff / Bad customer service ':2, 'Excess Documents Required':3}
data_group['Reason for exiting company'] = 
data_group['Reason for exiting company'].map(reason)

# Ordinal encoding of 'Gender'
Gender = {'Female':0, 'Male':1}
data_group['Gender'] = 
data_group['Gender'].map(Gender)

# Ordinal encoding of 'Geography'
Geography = {'France':0, 'Spain':1, 'Germany':2}
data_group['Geography'] = 
data_group['Geography'].map(Geography)


# ### Analyzing the data

# In[22]:


data_group.head()


# In[23]:


# showing correlation coefficients between variables

corrMatrix = 
data_group[['CreditScore','Gender','Geography','Age',
'Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Exited','Reason for exiting company']].corr()
fig, ax = 
plt.subplots(figsize=(10,10))  # Sample figsize in inches

# Visualize the correlation matrix
plt.title('Correlation Matrix of all our Features', pad=20, fontweight='bold') # pad gives the offset of the title from the top of the axes
sns.heatmap(corrMatrix, cmap='Blues', annot=True)


# In[24]:


data_group = data_group.dropna()
data_group.head()


# ### Data preparation

# In[25]:


x = data_group[['Tenure']]
y = data_group['Reason for exiting company']


# ### Splitting the dataset

# In[26]:


x_train, x_test, y_train, y_test = 
train_test_split(x, y, test_size=0.3, shuffle=False, random_state=0)


# ### Implementation of the model

# In[27]:


DecisionTreeRegModel2 = 
DecisionTreeRegressor()
DecisionTreeRegModel2.fit(x_train,y_train)
y_pred = DecisionTreeRegModel2.predict(x_test)


# In[28]:



print("The predicted values of exited are ", y_pred)


# In[29]:


print("The actual values of exited are ", y_test)


# ### Evaluating the Performance

# In[30]:


r2_score(y_test, y_pred)


# ### Decoding the encoded data

# In[31]:


res =  [round(abs(ele)) for ele in y_pred]


# In[32]:


df3 = y_test.to_frame()
df3['Reason for exiting company Predicted'] = res
df3['CustomerId'] = df['CustomerId']


# In[33]:


df3.head()


# In[34]:


df3["Reason for exiting company"] = 
df3["Reason for exiting company"].replace(999, "Not Exited", regex=True)
df3["Reason for exiting company Predicted"] = 
df3["Reason for exiting company"].replace(999, "Not Exited", regex=True)


# In[35]:


df3["Reason for exiting company"] = 
df3["Reason for exiting company"].replace(0, "High Service Charges/Rate of Interest", regex=True)
df3["Reason for exiting company Predicted"] =
df3["Reason for exiting company"].replace(0, "High Service Charges/Rate of Interest", regex=True)


# In[36]:


df3["Reason for exiting company"] = 
df3["Reason for exiting company"].replace(1, "Long Response Times", regex=True)
df3["Reason for exiting company Predicted"] = 
df3["Reason for exiting company"].replace(1, "Long Response Times", regex=True)


# In[37]:


df3["Reason for exiting company"] = 
df3["Reason for exiting company"].replace(2, "Inexperienced Staff / Bad customer service ", regex=True)
df3["Reason for exiting company Predicted"] = 
df3["Reason for exiting company"].replace(2, "Inexperienced Staff / Bad customer service ", regex=True)


# In[38]:


df3["Reason for exiting company"] = 
df3["Reason for exiting company"].replace(3, "Excess Documents Required", regex=True)
df3["Reason for exiting company Predicted"] = 
df3["Reason for exiting company"].replace(3, "Excess Documents Required", regex=True)


# In[39]:


df3


# In[40]:


type(df3)


# In[41]:


df3.head()


# ### Exporting the prediction data into Excel - .csv format

# In[42]:


df3.to_csv(r'C:\Users\sachi\sachi_bscit_Sem4_ipynb\CUSTOMER_CHURN_PREDICTION_FINAL_YEAR\pred.csv', index = False)


# In[43]:


df_new = df[df['Exited'] == 1]
sns.catplot(y="Reason for exiting company", hue="Tenure", kind="count", aspect=3.5, 
            data=df3)


# In[44]:


import pickle
pickl = {'model': DecisionTreeRegModel2}
pickle.dump(pickl, open('model_file2' + ".p", "wb"))
print("Model 2 pickled")


# ### Export the decision tree graph - graph description language format

# In[45]:


from sklearn.tree 
import export_graphviz 

export_graphviz(DecisionTreeRegModel1, out_file ='tree1.dot', 
feature_names =['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']) 


# In[ ]:

