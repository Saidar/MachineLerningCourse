import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

data = pd.read_csv("D:/Sai/JavaDoc/Cousera/5/2/gbm-data.csv")

y = data[data.columns[1]].values
x = data[data.columns[1:]].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

ls = [1, 0.5, 0.3, 0.2, 0.1]

for i in ls:
    clf = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=i);
    
    clf.fit(x_train, y_train)
    qual_test = clf.staged_decision_function(x_test)
    qual_train = clf.staged_decision_function(x_train)
    
    predict = clf.predict(x_test)
    
    pred_trans = 1 / (1 + math.exp(-predict))
    
    plt.figure()
    plt.plot(test_loss, 'r', linewidth=2)
    plt.plot(train_loss, 'g', linewidth=2)
    plt.legend(['test', 'train'])
    