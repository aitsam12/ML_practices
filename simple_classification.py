import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.neighbors import KNeighborsClassifier

# loading dataset
iris = datasets.load_iris()
#print(iris.DESCR)

#features and labels
features = iris.data
labels = iris.target
#print(features[0], labels[0])

# training the classifier
clf = KNeighborsClassifier()
clf.fit(features, labels)

# prediction
pred = clf.predict([[31,1,1,1]])
print(pred)
