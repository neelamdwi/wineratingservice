import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pickle
import sys

#load data
df = pd.read_csv("..\winequality-red.csv")

#split data
X = df['fixed_acidity'].values.reshape(-1, 1)
y = df['quality'].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#train model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#create pickle file
filename = 'regress_model'
pickle.dump(regressor, open(filename, 'wb'))

