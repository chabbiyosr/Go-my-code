
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data=pd.read_csv("titanic-passengers.csv",sep=';')


# In[3]:


data.head()


# In[4]:


data.isnull().sum()


# In[5]:


data["Cabin"].isnull().sum().sum()


# In[6]:


data["Cabin"].fillna('G6',inplace=True)


# In[7]:


data["PassengerId"].fillna('456',inplace=True)


# In[19]:


data["Age"].fillna('00',inplace=True)


# In[9]:


data.tail()


# In[10]:


data.isnull().sum()


# In[17]:


data["Embarked"].fillna('1',inplace=True)


# In[12]:


data.isnull().sum()


# In[13]:


#preparing data for logistic regression
data["Survived"]=data["Survived"].map({"Yes": 1, "No": 0})   #convert admitted variable into numerical


# In[14]:


data["Embarked"]=data["Embarked"].map({"C": 1, "S": 0})
data["Sex"]=data["Sex"].map({"female": 1, "male": 0})


# In[20]:


data


# In[21]:


#import relevant libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

#features extraction
x = data[['Age', 'Pclass','Embarked','Sex']]
y = data['Survived']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)  #splitting data with test size of 25%

logreg = LogisticRegression()   #build our logistic model
logreg.fit(x_train, y_train)  #fitting training data
y_pred  = logreg.predict(x_test)    #testing model’s performance
print("Accuracy={:.2f}".format(logreg.score(x_test, y_test)))


# In[25]:


import seaborn as sns
sns.regplot(x='Pclass',y='Survived',data=data)


# In[26]:


confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)


# In[27]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[38]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


# In[42]:


fpr, tpr, thresholds = roc_curve(y_test,y_pred)


# In[43]:


plot_roc_curve(fpr, tpr)


# **AUC - ROC curve**: *is a performance measurement for classification problem at various thresholds settings. ROC is a probability curve and AUC represents degree or measure of separability. It tells how much model is capable of distinguishing between classes*
