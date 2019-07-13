
import pandas as pd
from sklearn.svm import SVC

data = pd.read_csv("D:\Sai\JavaDoc\Cousera\svm-data.csv")
print(data)

svc = SVC(C=100000, kernel='linear', random_state=241)

svc.fit(data[data.columns[1:]], data['x'])

res = svc.support_vectors_
#print(res)

for i in range(len(res)):
    print(data[lambda x: x['y1'] == res[i][0]])


