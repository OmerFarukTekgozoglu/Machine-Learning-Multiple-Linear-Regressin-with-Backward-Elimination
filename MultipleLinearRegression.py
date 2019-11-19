# -*- coding: utf-8 -*-

"""
Created by
@author: Ömer Faruk
"""
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')
y = dataset.iloc[:,4].values
X = dataset.iloc[:,:-1].values

#Categorical to numerical transform
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_X = LabelEncoder()
X[:,3] = label_encoder_X.fit_transform(X[:,3])
one_hot_encoder = OneHotEncoder(categorical_features = [3])
X = one_hot_encoder.fit_transform(X).toarray()

X = X[:, 1:]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Training part
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# Prediction part
prediction = linear_reg.predict(X_test)

predict_xtrain = linear_reg.predict(X_train)

print("Modelin doğruluk skoru: % " + str(linear_reg.score(X_test, y_test)*100))
#%%
# Backwward elimination method for finding a best algorithm
""" Step - 1 importing the library """
import statsmodels.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#%%
""" Step - 2 finding highest P-Value"""
# If we print out the "summary_OLS" a summary and P-Value comes front of us. x2 is the highest P values it's means that we should drop-out it.
regressor_OLS.summary()
#%%
""" Step - 3 Extracting to highest P-Value variable from our dataset"""

# In this situation highest P-val is score to 2th variable 
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#%%

""" Still Continue to Step-3 """

# The highest P is score to 1th variable

X_opt = X[:, [0, 3, 4, 5]] 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#%%

# The highest P is score to 2th variable again  (the 4th index)

X_opt = X[:, [0, 3, 5]] 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#%%

# The highest P is score to 2th variable (the 5th index)

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#%% SO FAR SO GOOD
# Backward elimination is done, that means you can use only the 0 and 3th features for the training but the accuracy is still be good enough
# Let's find out.

X_train_BE = X_train[:,[0,3]] 
X_test_BE = X_test[:,[0,3]]
#BE means BackwardElimination 

y_train_BE = y_train # this is not change because they are the labels and we didnt do anything to labels.
y_test_BE = y_test

linear_regression_BE = LinearRegression()
linear_regression_BE.fit(X_train,y_train)

prediction_BE = linear_regression_BE.predict(X_test)

print("Multiple linear regression with backwart elimination method accuracy: " + str(linear_regression_BE.score(X_test,y_test)*100))

#%%
import matplotlib.pyplot as plt

f=plt.figure(figsize=(15,8))
cx = f.add_subplot(1,3,1)
cx.scatter(X_train[:,2], y_train, color='blue')
cx.plot(X_train[:,2], linear_reg.predict(X_train[:]), color='magenta', linewidth=0.75)
dx = f.add_subplot(1,3,2)
dx.scatter(X_train[:,3], y_train, color='purple')
dx.plot(X_train[:,3], linear_reg.predict(X_train[:]), color='magenta', linewidth=0.75)
ex = f.add_subplot(1,3,3)
ex.scatter(X_train[:,4], y_train, color='cyan')
ex.plot(X_train[:,4], linear_reg.predict(X_train[:]), color='magenta', linewidth=0.75)
f.savefig("multipleLinearRegression.png")
f.show()

