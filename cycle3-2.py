#!/usr/bin/env python
# coding: utf-8

# # Canadian Health Community Survey: cycle 3 - 2005
# 
# MA,
# Ruhid Mirzayev - 2020

#     Questions have been solved in this project.
#     
# 1. Average hospital stay and physisian visit in Canada in 2005.
# 2. Defining problem 1 by age groups( aged 30 and below, 30-64, and 65 and more). Does cancer varies by age groups?
# 3. Finding demand for healthcare. Do walking for exercise, age, income, affect demand for healthcare?
# 4. Defining cancer by using age, income and BMI( Body Mass Index)

# In[1]:


# Importing essential libraries of Python as below:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk

from pandas import Series, DataFrame
from matplotlib import rcParams

get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 5,4


# In[16]:


# Importing Data from Macbook:
address = ('/Users/ruhidmirzayev/Downloads/myportfolio/cycle_2005/cchs-82M0013-E-2005-c3-1-main-file_F1.csv')
df=pd.read_csv(address)


# In[17]:


df.columns


#  #### We need to choose variables we need
#  The variable names need to be defined as a understandable way. We edited their names from Variable Description Handbook

# In[18]:


# We need to choose 11 variables out of 1284
df= df[['DHHEGAGE','DHHE_SEX', 'CCCE_131','INCEGPER', 'HWTEGHTM','HWTEGWTK','HWTEGBMI','HCUE_01', 'HCUEG01A', 'HCUEG02A', 'PACE_1A' ]]
df.columns = ['age','sex', 'cancer','income','height', 'weight', 'BMI', 'overnight_patient', 'hosp_stay', 'physisian_visit', 'walking']
df.head()


# In[20]:


df.info()


# In[21]:


df.describe()


# The mean age is 8.7 and income is 21. Clearly, all patients can not stay 96 days exactly. Something is wrong with this data. We need to check variable description which comes with data when we download it from database. In the next cell, I am going to explain the chosen variables.

# ## Variable descriptions:
# 
# 
# 1. DHHEGAE - age 
#      
# 1 = 12-14, 
# 2 = 15-17, 
# 3 = 18-19, 
# 4 = 20-24, 
# 5 = 25-29, 
# 6 = 30-34, 
# 7 = 35-39, 
# 8 = 40-44, 
# 9 = 45-49, 
# 10 = 50-54, 
# 11 = 55-59, 
# 12 = 60-64, 
# 13 = 65-69, 
# 14 = 70-74, 
# 15 = 75-79, 
# 16 = 80 and more
# 
# 2. DHHE_SEX - sex
# 
# 1 - Male
# 2 - Female
# 
# 3. CCCE_131 - cancer
# 
# 1 - yes
# 2 - no
# 6 - not applicable
# 7 - do not know
# 8 - refusal
# 9 - not stated
# 
# 4. INCEGPER - Total houshold income
# 
# 1 =no income, 
# 2 = less than 15000, 
# 3 = 15000-29999, 
# 4 = 30000-49999, 
# 5 = 50000-79999, 
# 6 = 80000 or more. 
# 96 = not aplicable
# 97 = do not know
# 98 = refusal
# 99 = not stated
# 
# 5. HWTEGHTM - height in metres
# 
# 1.27 - 1.257 to 1.282
# 1.295 - 1.283 to 1.307
# ...
# ...
# Averages until 2.134
# But 9.999 - not stated
# 
# 6. HWTEGWTK - weight in kgs
# 
# 999.99 - not stated 
# 
# 7. HWTEGBMI - BMI(Body Mass Index)
# 
# 999.99 - Not Stated - Pregnant women
# 
# 
# 8. HCUE_01 - Overnight patient
# 
# 1 = yes
# 2 = no
# 6 = not applicable
# 7 = do not know
# 8 = refusal
# 9 = not stated
# 
# 9. If yes in HCUE_01
# 
# HCUEG01A - Number of nights as a patient
# 
# 1 = 1
# 2 = 2
# 3 = 3
# 
# ...
# 
# 30 = 30
# 31 = 31 and more
# 96 = not applicable (1 - Check Notes below)
# 97 = do not know
# 98 = refusal
# 99 = not stated (2 - Check notes below) 
# 
# 10. HCUEG02A - Physisian visit or number of consultations
# 1 = 1
# 2 = 2
# 3 = 3
# ...
# 
# 30 = 30
# 31 = 31 and more
# 96 = not applicable
# 97 = do not know
# 98 = refusal
# 99 = not stated
# 
# 11. PACE_1A - Last three month walking
# 
# 1 = yes
# 2 = no
# 6 = not applicable
# 7 = do not know
# 8 = refusal
# 9 = not stated (3 check Notes below)
# 
#     Notes:
# 
# (1) They said 'no' in HCUE_01. It implies that we need to replace them with 0.
# 
# (2) From description of variables guide,in HCUEG01A 99 = 1 night stay, this is because these people have not been questioned in HCUE_01. There are couple of previous variables which also define hospital stay or number of nights as a patient. Thus it is useless to ask again this question
# 
# (3) People did interview by proxy.

# ### Cleaning the data

# In[28]:


df.replace(to_replace = {
    'age': {1:13, 2:16, 3:18.5, 4:22, 5:27, 6:32, 7:37, 8:42, 9:47, 10:52, 11:57, 12:62, 13:67, 14:72, 15:77, 16:80},
    'sex': {2:0},
    'cancer': {2:0, 6: np.NaN, 7:np.NaN, 8: np.NaN, 9:np.NaN},
    'income': {1:0, 2:7500, 3:22500, 4:40000, 5:65000, 6:80000, 96: np.NaN, 97: np.NaN, 98: np.NaN, 99: np.NaN},
    'height': {9.999 : np.NaN},
    'weight': {999.99: np.NaN},
    'BMI':{999.99: np.NaN},
    'hosp_stay': {96:0, 97: np.NaN, 98: np.NaN, 99:1},
    'overnight_patient': {2:0, 7:np.NaN, 8:np.NaN, 9:np.NaN},
    'physisian_visit': {96:np.NaN, 97:np.NaN, 98:np.NaN, 99:np.NaN},
    'walking':{2:0, 6: np.NaN, 7: np.NaN, 8:np.NaN, 9: np.NaN}}, inplace = True)
df.head()


# In[32]:


df.describe()


# As we see above, income, age and so on variables look normal after replacing them.

# In[29]:


df.info()


# ## Dealing with null values
# 
# The number of null values are not considerable, that is why we can drop all the null values from our data.
# 
# The alternative is to replace null values with the mean of variables, but in our case some variable are catigorical and it will not make a sence if we do it.

# In[34]:


df = df.dropna()
df.info()


# In[76]:


df.describe()


# ### Question 1
# 
# #### Average hospital stay and physisian visit in Canada in 2005.

# In[37]:


df[['hosp_stay', 'physisian_visit']].describe()


# In[69]:


df.hist(column = 'physisian_visit')
df.hist(column = 'hosp_stay')


# ### Question 2
# 
# #### Defining hospital stay and physician visit by age groups
# 
# In order to find out the problem above by age groups, we need to create new variable which is catigorical and has only three values such as 1: 30 years and below, 2: 30 to 64 and 3: 65 and more.
# 

# In[38]:


df['age_groups'] = df['age'].replace([13,16,18.5,22,27,32,37,42,47,52,57,62,67,72,77,80], [1,1,1,1,1,2,2,2,2,2,2,2,3,3,3,3])
df.head()


# In[40]:


#Defining problem by age groups
df[['hosp_stay','physisian_visit', 'cancer']].groupby(df['age_groups']).mean()


# As we see from above,  If the age of  groups are increasing, cancer, hospital stay, and physisian visits rises as well. 

# ### Question 3
# 
# #### Linear regression
# Let's say healthcare demand is hospital stay. 
# 
# hosp_stay = b + income * b1 + walking * b2 + income * b3 + u. We need to find b1, b2 and b3.

# In[42]:


# First let'check Pearson correlation
df[['hosp_stay', 'age', 'income', 'walking']].corr()
# The results below do not show strong correlation


# In[75]:


import seaborn as sb
m = df[['income']]
n = df['hosp_stay']
sb.regplot(m,n, data = df)


# In[53]:


## Linear regressin by Stats Models
import statsmodels.api as sm


X = df[['age', 'income', 'walking']]
y = df['hosp_stay']

model = sm.OLS(y, X).fit()
predictions = model.predict(X)

model.summary()


# As we see p values are statistically significant, it enables us to say that if age increases hospital stay increases as well. On the contrary, there is negative correlation between income and hospital stay. The same patterns can be seen for walking variables.

# ### Question 4
# 
# Defining cancer by age, BMI and income
# 
# K nearest neighbor model is used

# In[54]:


# Defining dependent and target variables
X1 = df[['age', 'income', 'BMI']] # independent variables
y1 = df['cancer'] # Binary target variable


# #### Normalizing the data
# 
# This process gives data zero mean and unit variance

# In[56]:


# importing libraries
from sklearn import preprocessing


# In[59]:


X1 = preprocessing.StandardScaler().fit(X1).transform(X1.astype(float))
X1[0:5]


# #### Training the data
# 
# We will use train test split in order to make the best out of sample prediction.

# In[60]:


from sklearn.model_selection import train_test_split # the library
X1_train, X1_test, y1_train, y1_test = train_test_split( X1, y1, test_size=0.2, random_state=4)
print ('Train set:', X1_train.shape,  y1_train.shape)
print ('Test set:', X1_test.shape,  y1_test.shape)


# In[61]:


# Library fot k nearest
from sklearn.neighbors import KNeighborsClassifier
# Training
k = 4  # We choose random numbers for k value
n_gh = KNeighborsClassifier(n_neighbors = k).fit(X1_train,y1_train)
n_gh


# #### Predicting 

# In[62]:


y_hat = n_gh.predict(X1_test)
y_hat[0:5]


# #### Accuracy assessment

# In[63]:


from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y1_train, n_gh.predict(X1_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y1_test, y_hat))


# In[64]:


# What if we choose k = 6
k = 6
n_gh6 = KNeighborsClassifier(n_neighbors = k).fit(X1_train,y1_train)
y_hat6 = n_gh6.predict(X1_test)
print("Train set Accuracy: ", metrics.accuracy_score(y1_train, n_gh6.predict(X1_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y1_test, y_hat6))


# #### Choosing right value for K

# In[66]:


Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    n_gh_k = KNeighborsClassifier(n_neighbors = n).fit(X1_train,y1_train)
    y_hat_k=n_gh_k.predict(X1_test)
    mean_acc[n-1] = metrics.accuracy_score(y1_test, y_hat_k)

    
    std_acc[n-1]=np.std(y_hat_k == y1_test)/np.sqrt(y_hat_k.shape[0])

mean_acc


# In[67]:


plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()


# In[68]:


print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 


# In[ ]:




