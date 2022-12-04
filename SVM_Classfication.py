import pandas as pd
import numpy as np
import os
import glob
from sklearn import svm
from sklearn.ensemble import IsolationForest
import time

from sklearn.metrics import precision_score, accuracy_score, f1_score, roc_curve, auc,confusion_matrix,recall_score
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

import warnings 
warnings.filterwarnings(action= 'ignore')



df = pd.read_csv('./dataset.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


lsvc = LinearSVC()

lsvc.fit(X_train, y_train)

predictions = lsvc.predict(X_test)

target_names = ['no', 'yes']
print(" Results for Linear SVM classifier")

print(classification_report(y_test, predictions, target_names=target_names))









