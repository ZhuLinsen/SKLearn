import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing

data = pd.read_excel('Folds5x2_pp.xlsx')
x = data.iloc[:, :4]
y = data.iloc[:, 4]
model = KNeighborsRegressor()

x = preprocessing.scale(x)# 标准化
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=1)


model.fit(x_train,y_train)


print(model.score(x_test, y_test))
print('MSE:', mean_squared_error(y_test, y_predict))
print('RMSE:', math.sqrt(mean_squared_error(y_test, y_predict)))
