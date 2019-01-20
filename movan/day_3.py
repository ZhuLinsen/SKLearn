from sklearn import datasets
from sklearn.linear_model import LinearRegression

load_data = datasets.load_boston()
data_X = load_data.data
data_y = load_data.target

model = LinearRegression()
model.fit(data_X, data_y)

# print(model.predict(data_X[:4]))
# print(data_y[:4])
print(model.coef_) #y=kx+b的k
print(model.intercept_) #b
print(model.get_params)
print(model.score(data_X, data_y)) #R^2 coefficient of determination