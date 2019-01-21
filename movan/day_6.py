#交叉验证 cross validation

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X = iris.data
y = iris.target

import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    loss = -cross_val_score(knn, X, y,cv=10,scoring='neg_mean_squared_error')#for regression
    scores = cross_val_score(knn, X, y, cv=10, scoring = 'accuracy')#for classification
    k_scores.append(loss.mean())
print(k_scores)

#可视化
plt.plot(k_range, k_scores)
plt.xlabel('Value of K of KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()