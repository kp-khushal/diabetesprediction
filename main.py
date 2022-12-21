import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_diabetes
# from sklearn import load_boston
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import pickle


cancer = load_diabetes()

# print(cancer.DESCR)
# print(cancer.feature_names)

# load data set in dataframes
dataset = pd.DataFrame(cancer.data,columns=cancer.feature_names)

#  display head of dataset
# print(dataset.head())


#add new col [price] in dataset 
dataset['price'] = cancer.target
# print(dataset.head())

# check for null value and sum to null 
# print(dataset.isnull().sum())

# exploratory data analysis : 
# correlaton
# print(dataset.corr())

# Display Scatter Graph Plot 
# print(plt.scatter(dataset['price'],dataset['bmi']))

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

# print(x.head())
# print(y)
 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=40) 
# print(y_test)

scaler = StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

# model training

regression = LinearRegression()
regression.fit(x_train,y_train) 
# print(regression.coef_)
# print(regression.intercept_)
# print(regression.get_params())

# predict test data
reg_pred = regression.predict(x_test)
# print(reg_pred)
residuals = y_test-reg_pred
# print(residuals)
# print(mean_absolute_error(y_test,reg_pred))
# print(mean_squared_error(y_test,reg_pred))
# print(np.sqrt(mean_squared_error(y_test,reg_pred)))

score = r2_score(y_test,reg_pred)
# print(score)

1 - (1-score)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)

cancer.data[0].reshape(1,-1)

scaler.transform(cancer.data[0].reshape(1,-1))

# regression.predict(cancer.data[0].reshape(1,-1))

regression.predict(scaler.transform(cancer.data[0].reshape(1,-1)))

pickle.dump(regression,open('scaling.pkl','wb'))

pickle_model =  pickle.load(open('scaling.pkl','rb'))
print(regression.predict(cancer.data[0].reshape(1,-1)))


