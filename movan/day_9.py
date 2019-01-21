from sklearn import svm
from sklearn import datasets

clf = svm.SVC()
iris = datasets.load_iris()
X,y = iris.data, iris.target
clf.fit(X,y)

#保存
# method 1: pickle
import pickle

#method 2：joblib
from sklearn.externals import joblib
#Sve
joblib.dump(clf, 'save\\clf.pkl')
#restore
clf3 = joblib.load('save\\clf.pkl')
print(clf3.predict(X[0:1]))
