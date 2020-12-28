# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 02:04:50 2020

@author: shara
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
sal_train = pd.read_csv("F:\\Warun\\DS Assignments\\DS Assignments\\SVM\\SalaryData_Train(1).csv")
sal_test = pd.read_csv("F:\\Warun\\DS Assignments\\DS Assignments\\SVM\\SalaryData_Test(1).csv")
string_columns=["workclass","maritalstatus","occupation","relationship","race","sex","native"]

for col in string_columns:
    plt.figure(figsize = (11,6))
    sns.barplot(sal_train[col].value_counts(),sal_train[col].value_counts().index, data = sal_train)
    plt.title(col)
    plt.tight_layout()


number = preprocessing.LabelEncoder()
for i in string_columns:
    sal_train[i] = number.fit_transform(sal_train[i])
    sal_test[i] = number.fit_transform(sal_test[i])

sal_train.isin(['?']).sum(axis=0)
sal_train.info()
sal_test.isin(['?']).sum(axis=0)
sal_test.info()


column_list = []
iqr_list = []
out_low = []
out_up = []
tot_outlier = []

for i in sal_train.describe().columns : 
    QTR1 = sal_train[i].quantile(0.25)
    QTR3 = sal_train[i].quantile(0.75)
    IQR = QTR3 - QTR1
    LTV = QTR1 - (1.5* IQR)
    UTV = QTR3 + (1.5 * IQR)
    current_column = i
    current_iqr = IQR
    bl_LTV = sal_train[sal_train[i] < LTV][i].count()
    ab_UTV = sal_train[sal_train[i] > UTV][i].count()
    TOT_outliers = bl_LTV + ab_UTV
    column_list.append(current_column)
    iqr_list.append(current_iqr)
    out_low.append(bl_LTV)
    out_up.append(ab_UTV)
    tot_outlier.append(TOT_outliers)
    outlier_report = {"Column_name" : column_list, "IQR" : iqr_list, "Below_outliers" : out_low, "Above_outlier" : out_up, "Total_outliers" : tot_outlier}
    outlier_report = pd.DataFrame(outlier_report)
    print(outlier_report)

sns.boxplot(data = sal_train.age , orient = "n", palette = "Set3")
sns.boxplot(data = sal_train.workclass , orient = "n", palette = "Set3")
sns.boxplot(data = sal_train.education , orient = "n", palette = "Set3")
sns.boxplot(data = sal_train.educationno , orient = "n", palette = "Set3")
sns.boxplot(data = sal_train.maritalstatus , orient = "n", palette = "Set3")
sns.boxplot(data = sal_train.occupation , orient = "n", palette = "Set3")
sns.boxplot(data = sal_train.relationship , orient = "n", palette = "Set3")
sns.boxplot(data = sal_train.race , orient = "n", palette = "Set3")
sns.boxplot(data = sal_train.sex , orient = "n", palette = "Set3")
sns.boxplot(data = sal_train.capitalgain , orient = "n", palette = "Set3")
sns.boxplot(data = sal_train.capitalloss , orient = "n", palette = "Set3")
sns.boxplot(data = sal_train.hoursperweek , orient = "n", palette = "Set3")
sns.boxplot(data = sal_train.native , orient = "n", palette = "Set3")

f, ax = plt.subplots(figsize=(12,6))
sns.heatmap(sal_train.corr(), annot=True, fmt='.2f')



colnames = sal_train.columns
len(colnames[0:13])
trainX = sal_train[colnames[0:13]]
trainY = sal_train[colnames[13]]
testX  = sal_test[colnames[0:13]]
testY  = sal_test[colnames[13]]

x_train = trainX.drop(columns = ["education"])
x_test = testX.drop(columns=["education"])
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier

SGD = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000)
SGD.fit(x_train, trainY)

# model_linear = OneVsRestClassifier(SVC(kernel = "linear", cache_size=2000), n_jobs = -1)
# model_linear.fit(x_train,trainY)


pred_test_sgd = SGD.predict(x_test)
np.mean(pred_test_sgd==testY)
linear = classification_report(testY, pred_test_sgd)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)
SGD.fit(x_train_scaled, trainY)
pred_test_sgd_scaled = SGD.predict(x_test_scaled)
np.mean(pred_test_sgd_scaled==testY)
linear_scaled = classification_report(testY, pred_test_sgd_scaled)
print(linear_scaled)