from sklearn import datasets
from sklearn.linear_model import LinearRegression

load_data = datasets.load_boston()
data_X = load_data.data
data_y = load_data.target

model = LinearRegression()
model.fit(data_X, data_y)

print(model.predict(data_X[:4]))
print(data_y[:4])