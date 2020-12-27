
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt


# In[2]:


data=pd.read_csv("titanic-passengers.csv",sep=';')


# In[3]:


data.head()


# In[4]:


data.isnull().sum()


# In[5]:


data["Title"]=data["Name"].str.lower().str.extract('([a-z]*\.)', expand=True)


# In[6]:


avg_dr = data[((data["Title"]=="dr."))]
avg_master = data[((data["Title"]=="master."))] 
avg_miss = data[((data["Title"]=="miss."))] 
avg_mr = data[((data["Title"]=="mr."))] 
avg_mrs = data[((data["Title"]=="mrs."))] 


# In[7]:


lookfor = np.array(['mrs.','sir.','countess.', 'lady.', 'master.', 'miss.', 'mlle.', 'mme.','mrs.','ms.', 'sir.'])


# In[8]:


cleanup={"Survived":{"yes":1,"no":0}} 
data.replace(cleanup,inplace=True)
data[["Survived","Pclass"]].groupby(["Survived"], as_index=True).mean()


# In[9]:


data


# In[10]:


#data.rename(columns={'Name':'Title'},inplace= True )


# In[11]:


def plot_correlation_map( df ):

    corr = df.corr()

    s , ax = plt.subplots( figsize =( 12 , 10 ) )

    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

    s = sns.heatmap(

        Data.corr(), 

        cmap = cmap,

        square=True, 

        cbar_kws={ 'shrink' : .9 }, 

        ax=ax, 

        annot = True, 

        annot_kws = { 'fontsize' : 12 }

       )
    plt.show()
    plot_correlation_map(df)


# In[13]:


grid = sns.FacetGrid(data, row="Title", col="Pclass")
grid.map(sns.barplot,'Survived','SibSp')
grid.add_legend()


# In[14]:


data.SibSp.nunique()


# In[15]:


data.Title.nunique()


# In[17]:


data["tot_family_size"] = data["Parch"]+data["SibSp"]
print(data["tot_family_size"].mean(),data["tot_family_size"].std())
data["norm_family_size"] = (data["tot_family_size"]-data["tot_family_size"].mean())/(data["tot_family_size"].std())
data["norm_family_size"].hist()

