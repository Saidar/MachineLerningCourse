import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from time import time

# ***************************************************************************
# FIRST PART
# ***************************************************************************
# first
# import data
path = "D:/Study/Cousera/Python/Final/Data/"

data = "features.csv"
test = "features_test.csv"

df = pd.read_csv(path + data)
df_test = pd.read_csv(path + test)

# print(df.head())
# print(df_test.head())

# divide for x and y
x = df[df.columns[1:103]]
y = df[df.columns[104]]

# second
# count gaps
for i in range(0, 102):
    count = x[x.columns[i]].count()
    if (count < 97230):
        print("Column: ", i, " Name: ", x.columns[i], " Amount: ", count)

# Output:
# Column:  82  Name:  first_blood_time  Amount:  77677
# Column:  83  Name:  first_blood_team  Amount:  77677
# Column:  84  Name:  first_blood_player1  Amount:  77677
# Column:  85  Name:  first_blood_player2  Amount:  53243
# Column:  86  Name:  radiant_bottle_time  Amount:  81539
# Column:  87  Name:  radiant_courier_time  Amount:  96538
# Column:  88  Name:  radiant_flying_courier_time  Amount:  69751
# Column:  93  Name:  radiant_first_ward_time  Amount:  95394
# Column:  94  Name:  dire_bottle_time  Amount:  81087
# Column:  95  Name:  dire_courier_time  Amount:  96554
# Column:  96  Name:  dire_flying_courier_time  Amount:  71132
# Column:  101  Name:  dire_first_ward_time  Amount:  95404

# explanation
# first_blood_time - according to the explanation of the columns there can be a hero without first blood in first 5 min
# of the game
# radiant_bottle_time/ dire_first_ward_time - the same for both side(radiant and dire)


# third
# fill N/A
x = x.fillna(0)

# fourth
# target variable is in the y variable and there is radiant_win column only

# fifth

kf = KFold( n_splits=5, shuffle=True, random_state=241)

j = [10,20,30]
cross = []
for k in j:
    print("Fitting with %d in" %k)
    t0 = time()
    clf = GradientBoostingClassifier(n_estimators=k, verbose=False, random_state=241)
    clf.fit(x, y)

    print("Fitted with %d in :" %k, time() - t0 )

    t0 = time()
    cross.append(np.mean(cross_val_score(clf, x, y, scoring="roc_auc", cv=kf, n_jobs=-1)))
    print("Crossvalidated in: ", time() - t0, "\n")


print("GradientBoosting:")
print(cross)

# cross validation shows that after increasing number of trees the increasing of the metric goes down
# it means that we can increase the number of trees, but it wont give us a huge advantage of doing it

# to speed up one can use more splits in KFold or take only part of the input data or decrease the depth of trees


# ***************************************************************************
# SECOND PART
# ***************************************************************************

x_save = x

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x = scaler.fit_transform(x, y)
kf = KFold(n_splits=5, shuffle=True, random_state=241)

# ***************************************************************************
#first

j = np.power(10.0, np.arange(-6, 6))
cross = []
for k in j:
    print("Fitting:")
    t0 = time()
    if(k < 0.1):
        clf = LogisticRegression(C=k, penalty="l2", solver="lbfgs", verbose=False, random_state=241)
    else:
        clf = LogisticRegression(C=k, penalty="l2", solver="lbfgs", max_iter=1000, verbose=False, random_state=241)
    clf.fit(x, y)

    print("Fitted with %f in :" %k, time() - t0 )

    t0 = time()
    cross.append((k, np.mean(cross_val_score(clf, x, y, scoring="roc_auc", cv=kf))))
    print("Crossvalidated in: ", time() - t0, "\n")


print("LogisticRegression first with differents C:")
print(cross)

# optimal is 1.0, 0.7154712012052329 then the difference goes down
delta = []
for i in range(0, len(cross) - 1):
    print(j[i] , cross[i + 1][1] - cross[i][1])

# ***************************************************************************
#second

cross_without = 0
x_new = pd.DataFrame(x)

x_bag_heroes = x_new # save normilized data
x_new = x_new.drop(x_new.columns[[1,2,10,18,26,34,42,50,58,66,74]], axis=1)

C = 1
print("Fitting without lobby, r1_hero...:")
t0 = time()
clf = LogisticRegression(C=C, penalty="l2",solver='lbfgs', max_iter=1000, random_state=241)
clf.fit(x_new, y)

print("Fitted in :", time() - t0 )

t0 = time()
cross_without = (np.mean(cross_val_score(clf, x_new, y, scoring="roc_auc", cv=kf)))
#print("Crossvalidated in: ", time() - t0, "\n")

print("LogisticRegression without heroes:")
print(cross_without)

# out put is 0.7155035977962558, better for 3.23965910229429e-05

# ***************************************************************************
#third

x = x_save # restore data back


col = [2,10,18,26,34,42,50,58,66,74]
all_indef = []
max = 0
for i in col:
    col_in = pd.unique(x[x.columns[i]])
    for v in col_in:
        if v not in all_indef:
            all_indef.append(v)
            if max < v:
                max = v

# different 108
# max enum 112


# ***************************************************************************
# fourth

X_pick = np.zeros((x.shape[0], max))

for i, match_id in enumerate(x.index):
    for p in range(5):
        X_pick[i, x.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, x.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1


x_new_pick = pd.DataFrame(X_pick)
x_new = x_bag_heroes
x_new = x_new.drop(x_new.columns[[1,2,10,18,26,34,42,50,58,66,74]], axis=1)

for i in range(0, len(X_pick[0] - 1)):
    x_new["bag_%d" % i] = x_new_pick[x_new_pick.columns[i]]






# ***************************************************************************
# fifth

# make LR with bag of heroes

cross_bag = []
kf = KFold(n_splits=5, shuffle=True, random_state=241)

print("LogisticRegression for different C: ")
for k in j:

    print("Fitting with a bag of words:")
    t0 = time()
    clf = LogisticRegression(C=k, penalty="l2", solver='lbfgs', max_iter=1000, random_state=241).fit(x_new, y)
    print("Fitted with %f in :" % k, time() - t0)

    t0 = time()
    cross_bag.append((k, np.mean(cross_val_score(clf, x_new, y, scoring="roc_auc", cv=kf))))
    print("Crossvalidated in: ", time() - t0, "\n")

print("Result LogisticRegression with heroes_bag: ")
print(cross_bag, "\n")

# optimal output with 1.0 is 0.7498061161267557


print("Predicting :")

# ***************************************************************************
# sixth

x_test = df_test
x_test = x_test.fillna(0)

col = [2,3,11,19,27,35,43,51,59,67,75]
all_indef = []
max = 0
for i in col:
    col_in = pd.unique(x_test[x_test.columns[i]])
    for v in col_in:
        if v not in all_indef:
            all_indef.append(v)
            if max < v:
                max = v


X_pick = np.zeros((x_test.shape[0], 112))

for i, match_id in enumerate(x_test.index):
    for p in range(5):
        X_pick[i, x_test.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, x_test.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1

x_new_pick = pd.DataFrame(X_pick)

# change x_test norm and delete col
x_test[x_test.columns[1:]] = scaler.transform(x_test[x_test.columns[1:]]) # norm
x_test = pd.DataFrame(x_test)
x_test = x_test.drop(x_test.columns[col], axis=1)

# then add the bag
for i in range(0, len(X_pick[0] - 1)):
    x_test["bag_%d" % i] = x_new_pick[x_new_pick.columns[i]]


# fit new last model to predict
clf = LogisticRegression(C=1, penalty="l2", solver='lbfgs', max_iter=1000, random_state=241).fit(x_new, y)

predict = clf.predict_proba(x_test[x_test.columns[1:]])

print(predict)

for i,l in predict:
    print(i,",",l)