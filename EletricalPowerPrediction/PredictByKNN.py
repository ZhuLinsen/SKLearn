import math

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split

data = pd.read_excel('Folds5x2_pp.xlsx')
x = data.iloc[:, :4]
y = data.iloc[:, 4]
model = KNeighborsRegressor()

#分割训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3)

model.fit(x_train, y_train)

y_predict = model.predict(x_test)
print(model.score(x_test, y_test))
print('MSE:', mean_squared_error(y_test, y_predict))
print('RMSE:', math.sqrt(mean_squared_error(y_test, y_predict)))

cross_predict = cross_val_predict(model, x, y, cv=10)
print('MSE: ', mean_squared_error(y, cross_predict))
print('RMSE: ', math.sqrt(mean_squared_error(y, cross_predict)))

fig, ax = plt.subplots()
ax.scatter(y_test, y_predict, label="Predict dot")
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'b-', lw=4, label="Measured Line")
plt.xlabel("Measured")
plt.ylabel("Predicted")
plt.legend(loc="best")
plt.show()