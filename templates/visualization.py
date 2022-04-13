#!/usr/bin/env python
# coding: utf-8

# # FINAL YEAR PROJECT - CUSTOMER CHURN PREDICTION
# 

# ### NAME: SACHI SHAH
# ### CLASS: TYIT
# ### ROLL NO: 522

# # 

# ### Importing Libraries

# In[79]:


import pandas 
as pd #data manipulation operations
import matplotlib.pyplot 
as plt # for plotting graphs
import seaborn 
as sns # for plotting graphs
import numpy as np 
from sklearn 
import metrics # for predictive analytics
from sklearn.tree 
import DecisionTreeRegressor
from sklearn.model_selection 
import train_test_split
from sklearn.metrics 
import r2_score,mean_squared_error # accuracy
from IPython.display 
import display
from sklearn.metrics 
import precision_score, recall_score, f1_score, accuracy_score


# ### Importing dataset

# In[80]:


# Read data in the excel file
df = 
pd.read_csv('Churn_Modelling.csv')
# First 5 rows of the dataset
df.head(5) 


# ### Data Preprocessing

# In[81]:


# First 5 rows of the dataset
df.head() 


# In[82]:


# Number of rows and columns in the dataset
df.shape


# In[83]:


# dataypes of the columns
df.dtypes


# In[84]:


# Check columns list and missing values
df.isnull().sum()


# In[85]:


# Get unique count for each variable
df.nunique()


# ### Analyzing the data

# In[86]:


# Pie chart depicting the percentage of churned and retained customers in the dataset

labels = 
'Exited', 'Retained'
sizes = 
[df.Exited[df['Exited']==1].count(), df.Exited[df['Exited']==0].count()]
explode = 
(0, 0.1) # only "explode" the 2nd slice
colors = 
['#ff9999','#66b3ff']
fig1, ax1 = 
plt.subplots(figsize=(10, 8))
ax1.pie(sizes,explode=
explode, labels=labels, autopct='%1.1f%%', colors=colors) #autopct display the percent value
plt.title("Percentage of Churned and Retained Customers")
plt.show()


# In[87]:
labels = 
'Exited', 'Retained'
sizes = 
[df.Exited[df['Exited']==1].count(), 
df.Exited[df['Exited']==0].count()]
explode = 
(0, 0.1) # only "explode" the 2nd slice
colors = 
['#ff9999','#66b3ff']
fig1, ax1 = 
plt.subplots(figsize=(10, 8))
ax1.pie(sizes,explode=explode, 
labels=labels, autopct='%1.1f%%', colors=colors) #autopct display the percent value
plt.title("Percentage of Churned and Retained Customers")
plt.show()


# categorical plot for depicting the churners based on geography
sns.catplot(y="Geography", 
hue="Exited", kind="count",data=df) #hue is used to encode the points with different colors

sns.catplot(y="Tenure", 
hue="Exited", kind="count",data=df) #hue is used to encode the points with different colors


# In[88]:


# categorical plot for depicting the churners based on gender
sns.catplot(y="Gender", 
hue="Exited", kind="count", data=df)


# In[89]:


# categorical plot for depicting the reason of leaving of the exited customers based on tenure
df_new = df[df['Exited'] == 1]
sns.catplot(y="Reason for exiting company", 
hue="Tenure", kind="count", aspect=3.3, data=df_new) # aspect gives the Orientation of the plot


# In[90]:


# creating a copy of the main dataframe
data_group=df


# In[91]:


# showing correlation coefficients between variables

corrMatrix = 
data_group[['CreditScore','Age','Tenure',
'Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Exited']].corr()
fig, ax = 
plt.subplots(figsize=(10,10))  # Sample figsize in inches

# Visualize the correlation matrix
plt.title('Correlation Matrix of Our Numerical Features', 
pad=20, fontweight='bold') # pad gives the offset of the title from the top of the axes
sns.heatmap(corrMatrix, cmap='Blues', annot=True)


# ## Model 1: Prediction whether the customer will exit or not

# ### Data preparation

# In[92]:


x = 
data_group[['CreditScore','Age','Tenure',
'Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']]
y = data_group['Exited']


# ### Splitting the dataset

# In[93]:


# shuffle false: shuffles the data before splitting
# random_state: return same results for each execution
x_train, x_test, y_train, y_test = 
train_test_split(x, y, test_size=0.3, 
shuffle=False, random_state=0)


# ### Implementation of the model

# In[94]:


model = DecisionTreeRegressor()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)


# ### Accuracy of the model

# In[95]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred, y_test)*100, " %")


# ### Evaluating the Performance

# In[96]:


# r2_score: used to evaluate the performance of a regression-based machine learning model
r2_score(y_test, y_pred)


# In[97]:


print("The predicted values of exited are ", y_pred)
import pickle
pickl = 
{'model': model}
with open('model_file1.pickle', 'wb') as modelFile:
    print("inside")
    pickle.dump(model, modelFile)
print("Model 1 pickled")


# In[98]:


print("The predicted values of exited are ", y_pred)


# In[99]:


print("The actual values of exited are ", y_test)


# In[100]:


print('Precision: %.3f' % precision_score(y_test, y_pred))


# In[101]:


print('Recall: %.3f' % recall_score(y_test, y_pred))


# In[102]:


print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))


# In[103]:


print(metrics.classification_report(y_test, y_pred))


# ### Evaluating the Performance

# In[31]:


# r2_score: used to evaluate the performance of a regression-based machine learning model
r2_score(y_test, y_pred)


# ### Accuracy of the model

# In[32]:


from sklearn.metrics 
import accuracy_score
print(accuracy_score(y_pred, y_test)*100)


# # 

# ## Model 2: Predicting the customers' reason of leaving

# ### Dealing with categorical values

# In[33]:


# Label encoding: each label is converted into an integer value
reason = 
{'High Service Charges/Rate of Interest':0, 'Long Response Times':1, 
'Inexperienced Staff / Bad customer service ':2, 'Excess Documents Required':3}
data_group['Reason for exiting company'] = 
data_group['Reason for exiting company'].map(reason)


# ### Analyzing the data

# In[34]:


data_group = 
data_group.dropna()


# ### Data preparation

# In[35]:


x = 
data_group[['Tenure']]
y = 
data_group['Reason for exiting company']


# ### Splitting the dataset

# In[36]:


x_train, x_test, y_train, y_test = 
train_test_split(x, y, test_size=0.3, shuffle=False, random_state=0)


# ### Implementation of the model

# In[37]:


model = DecisionTreeRegressor()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)


# ### Evaluating the Performance

# In[38]:


r2_score(y_test, y_pred)


# In[41]:


print(metrics.classification_report(y_test, y_pred))


# In[40]:


print("The predicted values of exited are ", y_pred)


# In[60]:


print("The actual values of exited are ", y_test)


# ### Evaluating the Performance

# In[61]:


r2_score(y_test, y_pred)


# In[108]:


# Classification metrics can't handle a mix of continuous and multiclass targets so will throw an error
from sklearn.metrics 
import accuracy_score
print(accuracy_score(y_pred, y_test)*100)


# ### Decoding the encoded data

# In[31]:


res =  
[round(abs(ele)) for ele in y_pred]


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


df3.head()


# In[40]:


# type(df3)


# In[41]:


# df3.head()


# ### Exporting the prediction data into Excel - .csv format

# In[42]:


df3.to_csv(r'C:\Users\sachi\sachi_bscit_Sem4_ipynb\CUSTOMER_CHURN_PREDICTION_FINAL_YEAR\pred.csv', index = False)


# In[43]:


import pickle
pickl = {'model': model}
with open('model_file2.pickle', 'wb') as modelFile:
    print("inside")
    pickle.dump(model, modelFile)
print("Model 2 pickled")


# ### Export the decision tree graph - graph description language format

# In[ ]:


from sklearn.tree 
import export_graphviz 

export_graphviz(DecisionTreeRegModel1, out_file ='tree1.dot', 
feature_names =
['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']) 


# In[ ]:





# In[ ]:




# python code


df = 
pd.read_csv('templates/churning_customers.csv')
sns.catplot(y="Geography", 
hue=
"Reason for exiting company", 
kind="count",data=df) #hue is used to encode the points with different colors
plt.savefig('static/geo-reason.png')
index = df.index
number_of_rows = len(index)
df2 = 
pd.read_csv('templates/all_customers.csv')
index2 = df2.index
number_of_rows2 = 
len(index2)
labels = 
'Exited', 'Retained'
sizes = 
[number_of_rows2, number_of_rows]
explode = (0, 0.1) # only "explode" the 2nd slice
colors = ['#ff9999','#66b3ff']
fig1, ax1 = 
plt.subplots(figsize=(10, 8))
ax1.pie
(sizes,explode=explode, labels=labels, 
autopct='%1.1f%%', colors=colors) #autopct display the percent value
plt.title("Percentage of Churned and Retained Customers")
# plt.legend() 
plt.savefig('static/pie.png')
return render_template('visualize.html')
