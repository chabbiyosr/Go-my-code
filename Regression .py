
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import metrics


# In[2]:


data=pd.read_csv("kc_house_data.csv",sep=',')


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.get_dtype_counts()


# In[6]:


sns.distplot(data['price'], color="r", kde=False)


# In[7]:


data.isnull().sum()


# In[8]:


data.id.nunique()


# In[9]:


data.condition.nunique()


# In[10]:


data.price.nunique()


# In[11]:


def plot_correlation_map( df ):

    corr = df.corr()

    s , ax = plt.subplots( figsize =( 12 , 10 ) )

    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

    s = sns.heatmap(

        corr, 

        cmap = cmap,

        square=True, 

        cbar_kws={ 'shrink' : .9 }, 

        ax=ax, 

        annot = True, 

        annot_kws = { 'fontsize' : 12 }

        )


# In[12]:


import matplotlib.pyplot as plt
plt.plot([1,2,3])
plt.subplot(211)
plot_correlation_map(data)


# In[13]:


sns.heatmap(data[['price','view','grade','bedrooms','bathrooms','sqft_living','sqft_above','lat','sqft_basement']].corr(),annot=True)


# In[14]:


data['price'].corr(data['grade'])


# In[15]:


import matplotlib.pyplot as plt
plt.scatter(data["grade"],data["price"], color="r")
plt.title("grade Vs price ")
plt.ylabel("price")
plt.xlabel("grade");


# In[16]:


sns.boxplot("grade","price",data=data)


# In[17]:


sns.boxplot("waterfront","price",data=data)
plt.title("waterfront Vs price ")
plt.ylabel("price")
plt.xlabel("waterfront")


# In[18]:


sns.boxplot("view","price",data=data)
plt.title("view Vs price ")
plt.ylabel("price")
plt.xlabel("view")


# In[19]:


sns.stripplot(x=data["view"], y=data["price"],jitter=True)


# In[20]:


grid = sns.FacetGrid(data, row="waterfront", col="view")
grid.map(sns.barplot,'floors','grade')
grid.add_legend()


# In[21]:


Y = data["price"]
X = data[['grade','sqft_living']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 30)


# In[22]:


lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)


# In[23]:


plt.scatter(X_train.sqft_living,Y_train,  color='green')


# In[24]:


model=LinearRegression() 
model.fit(X_train,Y_train)
predicted=model.predict(X_test)
print("MSE", mean_squared_error(Y_test,predicted))
print("R squared", metrics.r2_score(Y_test,predicted))


# In[25]:


import matplotlib.pyplot as plt
plt.scatter(X.sqft_living,Y,color='r')
plt.title("Linear Regression")
plt.ylabel("price")
plt.xlabel("sqft_living")
plt.plot(X,model.predict(X),color='k')
plt.show


# In[26]:


x=data[["sqft_living","sqft_above"]]  #we have more than one input
y=data["price"].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=40) #splitting data with test size of 35%
model=LinearRegression() #build linear regression model
model.fit(X_train,Y_train) #fitting the training data
predicted=model.predict(X_test) #testing our model’s performance

print("MSE", mean_squared_error(Y_test,predicted))
print("R squared", metrics.r2_score(Y_test,predicted))


# In[27]:


plt.plot(x_train,y_train, color='black', linewidth=3)


# In[28]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures 
x= data[["sqft_living","sqft_above"]]
y= data["price"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=40)  #splitting data
lg=LinearRegression()
poly=PolynomialFeatures(degree=3)
x_train_fit = poly.fit_transform(x_train) #transforming our input data
lg.fit(x_train_fit, y_train)
x_test_ = poly.fit_transform(x_test)
predicted = lg.predict(x_test_)
print("MSE: ", metrics.mean_squared_error(y_test, predicted))
print("R squared: ", metrics.r2_score(y_test,predicted))


# In[29]:


x= data["sqft_living"].values.reshape(-1,1)
y= data["price"].values
poly = PolynomialFeatures(degree = 2) 
x_poly = poly.fit_transform(x) 
poly.fit(x_poly, y) 
lg=LinearRegression()
lg.fit(x_poly, y) 

plt.scatter(x, y, color="b")
plt.title("Linear regression")
plt.ylabel("price")
plt.xlabel("sqft_living")
plt.plot(x, lg.predict(poly.fit_transform(x)), color="r") 


# **Question2:**
# *The most important features are grade,Sqft_living,Sqft_above, due to the coorelation function, we can choose our features.*
