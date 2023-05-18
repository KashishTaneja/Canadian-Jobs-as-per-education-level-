#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import pyodbc


# In[2]:


conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=LAPTOP-BKNV5237;'
                      'Database=CanadianJobs;'
                      'Trusted_Connection=yes;')
cursor = conn.cursor()
cursor.execute('SELECT * FROM v0913_05')

for i in cursor:
    print(i)


# In[4]:


import pandas as pd
import pyodbc 

conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=LAPTOP-BKNV5237;'
                      'Database=CanadianJobs;'
                      'Trusted_Connection=yes;')

Can_Jobs = pd.read_sql_query('SELECT * FROM v0913_05', conn)


# In[5]:


Can_Jobs


# In[6]:


Male_df = Can_Jobs[['YEAR','Geography','Type_of_work','Wages','Education_level','Age_group','Wages_Male(in $)']]
Male_df['sex'] = ['Male'] * len(Male_df)


# In[7]:


Male_df_2 = Male_df.rename(columns={"Wages_Male(in $)": "Wages(in $)"})


# In[8]:


Female_df = Can_Jobs[['YEAR','Geography','Type_of_work','Wages','Education_level','Age_group','Wages_Female(in $)']]
Female_df['sex'] = ['Female'] * len(Female_df)
Female_df


# In[9]:


Female_df_2= Female_df.rename(columns={"Wages_Female(in $)": "Wages(in $)"})


# In[10]:


New_Can_Jobs= pd.concat([Male_df_2,Female_df_2])


# In[11]:


New_Can_Jobs_2= New_Can_Jobs.round(decimals=2)


# In[12]:


New_Can_Jobs_2


# In[13]:


New_Can_Jobs_3 =New_Can_Jobs_2.loc[New_Can_Jobs_2['Type_of_work'] != 'Both full- and part-time']


# In[14]:


New_Can_Jobs_4 = New_Can_Jobs_3.loc[New_Can_Jobs_3['Wages'] == 'Average weekly wage rate']


# In[15]:


New_Can_Jobs_5 = New_Can_Jobs_4.loc[New_Can_Jobs_4['Education_level'] != 'Total, all education levels']


# In[16]:


New_Can_Jobs6 = New_Can_Jobs_5.loc[New_Can_Jobs_5['Geography'] != 'Canada']


# In[17]:


New_Can_Jobs6.reset_index(drop=True, inplace=True)


# In[18]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[19]:


plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(New_Can_Jobs6['Wages(in $)'])
plt.show()


# In[20]:


Q_W_25, Q_W_75 = np.percentile(New_Can_Jobs6['Wages(in $)'], [25,75])
IQR_W = Q_W_75- Q_W_25
IQR_W


# In[21]:


upper_Wages = Q_W_75+(1.5*IQR_W)
upper_Wages


# In[22]:


lower_Wages = Q_W_25-(1.5*IQR_W)
lower_Wages


# In[23]:


New_Can_Jobs7 = New_Can_Jobs6[(New_Can_Jobs6['Wages(in $)'] < upper_Wages ) & (New_Can_Jobs6['Wages(in $)'] > lower_Wages)]


# In[24]:


plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(New_Can_Jobs7['Wages(in $)'])
plt.show()


# In[32]:


New_Can_Jobs8 = New_Can_Jobs7.drop(['Wages'], axis=1)


# In[33]:


New_Cana_Jobs8 = New_Can_Jobs8.rename(columns={"Wages(in $)": "Avg_Weekly_Wages(in $)"})


# In[34]:


New_Can_Jobs8.to_csv('file_name.csv', sep=',')


# In[69]:


new_df.to_csv('prediction_1.csv', sep=',')


# In[36]:


one_hot_1 = pd.get_dummies(New_Cana_Jobs8['Geography'])
New_Can_Jobs9 = New_Cana_Jobs8.drop('Geography',axis = 1)
New_Can_Jobs10 = New_Can_Jobs9.join(one_hot_1)


# In[37]:


one_hot_2 = pd.get_dummies(New_Can_Jobs10['Education_level'])
New_Can_Jobs11 = New_Can_Jobs10.drop('Education_level',axis = 1)
New_Can_Jobs12 = New_Can_Jobs11.join(one_hot_2)


# In[38]:


one_hot_3 = pd.get_dummies(New_Can_Jobs12['Age_group'])
New_Can_Jobs13 = New_Can_Jobs12.drop('Age_group',axis = 1)
New_Can_Jobs14 = New_Can_Jobs13.join(one_hot_3)


# In[39]:


dummy = pd.get_dummies(New_Can_Jobs14['sex'])
dummy_1 = pd.concat((dummy,New_Can_Jobs14), axis = 1)
dummy_2 = dummy_1.drop((['sex']),axis =1)
New_Can_Jobs15 = dummy_2.drop((['Male']),axis =1)
New_Can_Jobs16 = New_Can_Jobs15.rename(columns={"Female": "Sex"})


# In[40]:


dummy_3 = pd.get_dummies(New_Can_Jobs16['Type_of_work'])
dummy_4 = pd.concat((dummy_3,New_Can_Jobs16), axis = 1)
dummy_5 = dummy_4.drop((['Type_of_work']),axis =1)
dummy_6 = dummy_5.drop(['Full-time '] ,axis=1 )
New_Can_Jobs17 = dummy_6.rename(columns={'Part-time ': "Type_of_work"})


# In[41]:


New_Can_Jobs17.corr()


# In[42]:


import seaborn as sns 
plt.figure(figsize=(25,10))
sns.barplot(x="Geography",y="Avg_Weekly_Wages(in $)",data=New_Cana_Jobs8).set(title='Geography & Avg_Weekly_Wages')


# In[43]:


New_Cana_Jobs8.to_csv('Dataset')


# In[44]:


New_Can_Jobs17.to_csv('NewData')


# In[45]:


Education= pd.read_csv(r"C:\Users\Kashsih Taneja\Desktop\education.csv")
import seaborn as sns
plt.figure(figsize=(12,10))
sns.scatterplot('Education_Level', 'Avg_Weekly_Wages(in $)', data=Education,color = 'red')
plt.title("Avg_Weekly_Wages(in $) by education_Level")
plt.show()


# In[46]:


year= pd.read_csv(r"C:\Users\Kashsih Taneja\Desktop\Year.csv")
year.plot.line(x='Year', y='Avg_Weekly_Wages(in $)',marker = 'o',markerfacecolor= 'green')
plt.ylabel('Wages')
plt.title('Avg_Weekly_Wages & Year')
plt.show()


# In[47]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics


# In[48]:


New_Can_Jobs17.describe()


# In[49]:


y = New_Can_Jobs17["Avg_Weekly_Wages(in $)"]


# In[50]:


X = New_Can_Jobs17.loc[:, New_Can_Jobs17.columns != "Avg_Weekly_Wages(in $)"]


# In[51]:


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size= 0.2, random_state=0)


# In[52]:


lr = LinearRegression().fit(X_train,y_train)
cv_scores = cross_val_score(lr,X_train,y_train,cv=3)


# In[53]:


cv_scores


# In[54]:


y_pred = lr.predict(X_test)


# In[55]:


new_df = pd.DataFrame({'Actual' : y_test,'Predicted' : y_pred})


# In[56]:


new_df.head()


# In[66]:


new_col= pd.read_csv(r"C:\Users\Kashsih Taneja\Desktop\NewData.csv")


# In[67]:


y_pred_new = lr.predict(new_col)


# In[68]:


y_pred_new


# In[51]:


from sklearn.ensemble import RandomForestRegressor


# In[52]:


regr = RandomForestRegressor().fit(X_train,y_train)


# In[53]:


cv_scores_1 = cross_val_score(regr,X_train,y_train,cv=3)


# In[54]:


cv_scores_1


# In[55]:


y_pred_1 = regr.predict(X_test)


# In[56]:


new_df_1 = pd.DataFrame({'Actual' : y_test,'Predicted' : y_pred_1})


# In[57]:


new_df_1


# In[58]:


import numpy as np
from sklearn.tree import DecisionTreeRegressor


# In[59]:


regr_1 = DecisionTreeRegressor().fit(X_train,y_train)


# In[60]:


cv_scores_2 = cross_val_score(regr_1,X_train,y_train,cv=3)


# In[61]:


cv_scores_2


# In[62]:


y_pred_2 = regr_1.predict(X_test)


# In[63]:


new_df_2 = pd.DataFrame({'Actual' : y_test,'Predicted' : y_pred_2})


# In[64]:


new_df_2


# In[ ]:





# In[ ]:




