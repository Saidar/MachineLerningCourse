import pandas as pd
import numpy as np
import heapq as h
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.svm import SVC

newgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism','sci.space'])
x = newgroups.data
y = newgroups.target

print("Transf")
tfdv = TfidfVectorizer()
data = tfdv.fit_transform(x)
features = tfdv.get_feature_names()

#CV KFold
print("Grid")
grid = {'C':np.power( 10.0, np.arange(-5, 6))}
cv = KFold (n_splits=5, shuffle=True , random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf , grid , scoring='accuracy', cv=cv, n_jobs=-1)
gs.fit(data,y)

print("Est with best C")
clf2 = SVC(kernel='linear', random_state=241,C = gs.best_estimator_.C)
clf2.fit(data,y)

print("Abs of coef")
coef_abs = np.abs(clf2.coef_.data)

print("Take top 10")
top = h.nlargest(10, range(len(coef_abs)), coef_abs.take)
indexes = clf2.coef_.indices[top]

ans = []
for a in indexes:
    ans.append(features[a])

print(ans)