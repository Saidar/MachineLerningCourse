import pandas as pd
import numpy as np
from sklearn.metrics.classification import confusion_matrix, accuracy_score,\
    precision_score, recall_score, f1_score

data = pd.read_csv("D:/Sai/JavaDoc/Cousera/3/3_4/classification.csv")

data[np.isnan(data)] = 0
#data = data[np.isfinite(data['pred'])]
print(data)

#first
print("first:")
cnm = confusion_matrix(data['true'], data['pred'])
mat = np.matrix( [[cnm[1][1], cnm[0][1]] , [cnm[0][0], cnm[1][0]]])
print(mat, "\n")

# second 

print("second:")
acc = accuracy_score(data['true'], data['pred'])
print("accuracy: ",round(acc,2))

per = precision_score(data['true'], data['pred'])
print("percision: ",round(per,2))

rec = recall_score(data['true'], data['pred'])
print("recall: ", round(rec,2))

f_m = f1_score(data['true'], data['pred'])
print("f-metr: ", round(f_m,2), "\n")


#third
print("\nthird:")
from sklearn.metrics import roc_auc_score

data = pd.read_csv("D:/Sai/JavaDoc/Cousera/3/3_4/scores.csv")

metr = 0.0
max = [0, ""]
for i in range(1,5):
    metr = roc_auc_score(data['true'], data[data.columns[i]])
    #print(metr)
    if(metr > max[0]):
        max[0] = metr
        max[1] = data.columns[i]
print("max: ", max[1], " value: ", max[0])

#forth

from sklearn.metrics import precision_recall_curve
recall = []
for k in range(1,5):
    recall.append(np.array(precision_recall_curve(data['true'], data[data.columns[k]])))
    matr = np.array(precision_recall_curve(data['true'], data[data.columns[k]]))
    n_ar = [] # store >0.7
    tmp = [] # store colums [1]
    max = 0
    for j in range(0, len(matr[1])):
        tmp.append(matr[1][j])
        if(matr[1][j] >= 0.7):
            n_ar.append(matr[][j])
            if(matr[0][j] >= max):
                max = matr[0][j] 
                print(max)
            #print(j)
    print("Good Max: ",max)
    print("n_ar", n_ar)
    print("tmp", tmp)
#print(recall)
        
                   

    
    
    
