import pandas as pd
import numpy as np
from pandas import *

data = pd.read_csv('D:/Study/COUSERA/Python/1/titanic.csv', index_col='PassengerId')
print(data.head())
print(data.dtypes)

#1 amount of male and female
data['Sex'].value_counts();

print('1', data.groupby('Sex').size())

#2

survived = data.query('Survived == "1"').Survived.count()
f_sur = float(survived)
m = data['Sex'].count()

print('2',round(f_sur / m * 100 , 2))

#3
pclass = float(data.query('Pclass == 1').Pclass.count())
f_pass = data['Pclass'].count()
p_pclass = round( pclass / f_pass * 100 , 2)

print('3',p_pclass)

#4

mean = data['Age'].mean()

median = data['Age'].median()

print('4',round(mean,2),median)

#5

corr_coef = round(np.corrcoef(data['SibSp'], data['Parch'])[0,1], 2)


print('5',corr_coef)

#6

data['Split'] = data['Name'].str.split(' ');

#print(data['Split'])

idx = Int64Index([range(1,200)])
new_list = []

for ind in data['Split']:
    lenli = len(ind)
    for i, jnd in enumerate(ind):
        if(jnd == "Miss."):
            nextel = ind[(i + 1) % lenli]
            new_list.append(nextel)

names = ['Elizabeth', 'Laina', 'Ellen', 'Elizabeth']

for ind in new_list:
    names.append(str(ind).replace('{[()]}', ''))

new_names = [[1] * 2]

print(names)
print(new_names)


for i, ind  in enumerate(names):
    for j, jnd in enumerate(new_names):
        tmp1 = names[i]
        tmp2 = new_names[j][0]
        if(tmp1 == tmp2):
            new_names[j][1] = int(new_names[j][1]) + 1
            break
        elif (j == len(new_names) - 1 ):
            new_names.append([ind,1])
            break

print(new_names)