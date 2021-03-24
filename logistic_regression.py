import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
#print(list(iris.keys()))  
#print(iris.data)
#print(iris.data.shape) #shape of dataset
#print(iris.target)
#print(iris['DESCR'])

# Train a logistic regression to predict flower is verginica or not
X = iris['data'][:,3:]  # feature
#print(X)
y = (iris['target'] == 2).astype(np.int) # if target is 2 (verginica) then its True and 'astype(np.int)' will say it as 1.
#print(y)

# Train a logistic regression classifier
clf = LogisticRegression()
clf.fit(X,y)

# example prediction
example = clf.predict(([[3.6]]))  
#print(example)

# Plotting
X_new = np.linspace(0,3,1000).reshape(-1,1) # reshape(-1,1) will give many rows and one column, which is desired
y_prob = clf.predict_proba(X_new)
#print(y_prob[:,1]) 
plt.plot(X_new, y_prob[:,1], 'g-', label='Verginica')
plt.legend()
plt.show()

