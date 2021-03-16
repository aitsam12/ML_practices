import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import  mean_squared_error


diabeties = datasets.load_diabetes()
#print(diabeties.keys())
# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])
#print(diabeties.DESCR)

diabeties_X = diabeties.data[:, np.newaxis, 3] # making array of arrays

diabeties_X_train = diabeties_X[:-30] 
diabeties_X_test = diabeties_X[-30:] 

diabeties_Y_train = diabeties.target[:-30]
diabeties_Y_test = diabeties.target[-30:]

#calling model
model = linear_model.LinearRegression()

# data fitting
model.fit(diabeties_X_train, diabeties_Y_train)

# predict
diabeties_Y_predict = model.predict(diabeties_X_test)

print('mean squared error = {}'.format(mean_squared_error(diabeties_Y_test, diabeties_Y_predict)))
print('weights = {}'.format(model.coef_))
print('intercept = {}'.format(model.intercept_))

plt.scatter(diabeties_X_test, diabeties_Y_test)
plt.plot(diabeties_X_test, diabeties_Y_predict)

plt.show()