# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 21:56:30 2020

@author: shara
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import seaborn as sns
forest = pd.read_csv("F:\\DS Assignments\\SVM\\forestfires.csv")
forest1 = forest.drop(columns=['month', 'day'])
# forest1.loc[forest1.size_category=="small", "size_category"] = 0
# forest1.loc[forest1.size_category=="large", "size_category"] = 1
x = forest1.drop(columns = ["size_category"])
y = forest1["size_category"]
forest1["size_category"].value_counts()
forest1.head()
forest1.describe()
forest1.columns
sns.pairplot(data=forest1)
from sklearn.svm import SVC
x = forest1.drop(columns = ["size_category"])
y = forest1["size_category"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
SVC
model_linear = SVC(kernel = "linear")
model_linear.fit(x_train,y_train)
pred_test_linear = model_linear.predict(x_test)
np.mean(pred_test_linear==y_test)
from sklearn.metrics import classification_report
linear = classification_report(y_test, pred_test_linear)


# model_poly = SVC(kernel = "poly")
# model_poly.fit(x_train,y_train)
# pred_test_poly = model_poly.predict(x_test)
# np.mean(pred_test_poly==y_test)
# poly = classification_report(y_test, pred_test_poly)

# model_rbf = SVC(kernel = "rbf")
# model_rbf.fit(x_train,y_train)
# pred_test_rbf = model_rbf.predict(x_test)
# np.mean(pred_test_rbf==y_test)
# rbf = classification_report(y_test, pred_test_rbf)
