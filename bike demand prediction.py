import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

#importing the dataset
bikes = pd.read_csv('hour.csv')
bikes_cpy=bikes.copy()
#dropping the columns with no effect
bikes_cpy=bikes_cpy.drop(['index','date','casual','registered'],axis=1)
#checking for any null values in the dataset
bikes_cpy.isnull().sum()
bikes_cpy.hist(rwidth=0.8)
plt.tight_layout()
#subplotting the independent variables to find a realtion with the dependant variable
plt.subplot(2,2,1)
plt.title('Temparature vs demand')
plt.scatter(bikes_cpy['temp'],bikes_cpy['demand'],s=2)

plt.subplot(2,2,2)
plt.title('atemp vs demand')
plt.scatter(bikes_cpy['atemp'],bikes_cpy['demand'],s=2,c='g')

plt.subplot(2,2,3)
plt.title('Humidity vs demand')
plt.scatter(bikes_cpy['humidity'],bikes_cpy['demand'],s=2,c='b')

plt.subplot(2,2,4)
plt.title('windspeed vs demand')
plt.scatter(bikes_cpy['windspeed'],bikes_cpy['demand'],s=2,c='y')
plt.tight_layout()

colours=['g','b','y','r']
plt.subplot(3,3,1)
plt.title('average demand for each season')
cat_list = bikes_cpy['season'].unique()
cat_average= bikes_cpy.groupby('season')['demand'].mean()
plt.bar(cat_list,cat_average, color= colours)

plt.subplot(3,3,2)
plt.title('average demand for each month')
cat_list = bikes_cpy['month'].unique()
cat_average= bikes_cpy.groupby('month')['demand'].mean()
plt.bar(cat_list,cat_average, color= colours)

plt.subplot(3,3,3)
plt.title('average demand for each year')
cat_list = bikes_cpy['year'].unique()
cat_average= bikes_cpy.groupby('year')['demand'].mean()
plt.bar(cat_list,cat_average, color= colours)

plt.subplot(3,3,4)
plt.title('average demand for each hour')
cat_list = bikes_cpy['hour'].unique()
cat_average= bikes_cpy.groupby('hour')['demand'].mean()
plt.bar(cat_list,cat_average, color= colours)

plt.subplot(3,3,5)
plt.title('average demand for each weekday')
cat_list = bikes_cpy['weekday'].unique()
cat_average= bikes_cpy.groupby('weekday')['demand'].mean()
plt.bar(cat_list,cat_average, color= colours)

plt.subplot(3,3,6)
plt.title('average demand for each workingday')
cat_list = bikes_cpy['workingday'].unique()
cat_average= bikes_cpy.groupby('workingday')['demand'].mean()
plt.bar(cat_list,cat_average, color= colours)

plt.subplot(3,3,7)
plt.title('average demand for each weather')
cat_list = bikes_cpy['weather'].unique()
cat_average= bikes_cpy.groupby('weather')['demand'].mean()
plt.bar(cat_list,cat_average, color= colours)

plt.subplot(3,3,8)
plt.title('average demand for each holiday')
cat_list = bikes_cpy['holiday'].unique()
cat_average= bikes_cpy.groupby('holiday')['demand'].mean()
plt.bar(cat_list,cat_average, color= colours)

plt.tight_layout()

bikes_cpy['demand'].describe()
bikes_cpy['demand'].quantile([0.05,0.1,0.15,0.9,0.95,0.99])

correlation = bikes_cpy[['temp','atemp','humidity','windspeed','demand']].corr()

bikes_cpy=bikes_cpy.drop(['weekday','year','workingday','atemp','windspeed'],axis=1)
df1=pd.to_numeric(bikes_cpy['demand'],downcast='float')
plt.acorr(df1,maxlags=12)

df1=bikes_cpy['demand']
df2=np.log(df1)

plt.figure()
df1.hist(rwidth=0.9, bins=20)
plt.figure()
df2.hist(rwidth=0.9, bins=20)

bikes_cpy['demand'] = np.log(bikes_cpy['demand'])

t_1 = bikes_cpy['demand'].shift(+1).to_frame()
t_1.columns =['t-1']

t_2 = bikes_cpy['demand'].shift(+2).to_frame()
t_2.columns =['t-2']

t_3 = bikes_cpy['demand'].shift(+3).to_frame()
t_3.columns =['t-3']

bikes_cpy_lag = pd.concat([bikes_cpy,t_1,t_2,t_3],axis=1)
bikes_cpy_lag = bikes_cpy_lag.dropna()

bikes_cpy_lag.dtypes
bikes_cpy_lag['seasons'] = bikes_cpy_lag['seasons'].astype('category')
bikes_cpy_lag['holiday'] = bikes_cpy_lag['holiday'].astype('category')
bikes_cpy_lag['weather'] = bikes_cpy_lag['weather'].astype('category')
bikes_cpy_lag['month'] = bikes_cpy_lag['month'].astype('category')
bikes_cpy_lag['hour'] = bikes_cpy_lag['hour'].astype('category')

bikes_cpy_lag=pd.get_dummies(bikes_cpy_lag, drop_first=True)
#from sklearn.model_selection import train_test_split
Y = bikes_cpy_lag[['demand']]
X = bikes_cpy_lag.drop(['demand'],axis=1)

tr_size = 0.7 * len(X)
tr_size = int(tr_size)

X_train = X.values[0 : tr_size]
X_test = X.values[tr_size : len(X)]

Y_train = Y.values[0 : tr_size]
Y_test = Y.values[tr_size : len(Y)]

from sklearn.linear_model import LinearRegression

std_reg = LinearRegression()
std_reg.fit(X_train, Y_train)

r2_train = std_reg.score(X_train, Y_train)
r2_test = std_reg.score(X_test,Y_test)

Y_predict = std_reg.predict(X_test)

from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(Y_test, Y_predict))

Y_test_e = []
Y_predict_e = []
for i in range(0, len(Y_test)):
    Y_test_e.append(math.exp(Y_test[i]))
    Y_predict_e.append(math.exp(Y_predict[i]))

log_sq_sum=0.0

for i in range(0,len(Y_test_e)):
    log_a = math.log(Y_test_e[i]+ 1)
    log_p = math.log(Y_predict_e[i]+ 1)
    log_diff = (log_p - log_a)**2
    log_sq_sum = log_sq_sum + log_diff
    
rmsle = math.sqrt(log_sq_sum/len(Y_test))

print(rmsle)
