import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.datasets
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
dataset= sklearn.datasets.load_boston()
print(dataset)
   
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['price'] = dataset.target
df.shape
 df.head()

 correl=df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correl, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size
     <matplotlib.axes._subplots.AxesSubplot at 0x7ffabf14d2d0>
x=df.drop(['price'], axis=1)
y=df['price']
y_train,y_test,x_train,x_test=train_test_split(y,x,test_size=0.2,random_state=2)
mod=XGBRegressor()
mod.fit(x_train,y_train)
    
                   reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                  silent=None, subsample=1, verbosity=1)
pre=mod.predict(x_train)
print(pre)
   
pre=mod.predict(x_test)
s1=metrics.r2_score(y_test,pre)
s2=metrics.mean_absolute_error(y_test,pre)
print("R Squared Error:",s1)
print("Mean Absolute Error:",s2)
     
