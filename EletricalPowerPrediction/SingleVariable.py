import pandas as pd
from sklearn.datasets import load_iris

data = pd.read_excel('C:\\Users\\12259\\Desktop\\test.xlsx')
#print(data)

x = data.iloc[:, :4]
y = data.iloc[:, 4]
# print(type(x), type(y))

iris = load_iris()
print(type(iris))
