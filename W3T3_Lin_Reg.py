import math as m
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.linalg._fblas import izamax

data = pd.read_csv("D:/Sai/JavaDoc/Cousera/3/3_3/data-logistic.csv")

x_0 = np.array(data[data.columns[1]])
x_1 = np.array(data[data.columns[2]])
y = np.array(data[data.columns[0]])

k = 0.1




# x, y array; l - iterations of sum; n - which w1 or w2; w1 and w2 weights; c - regular
def w_up(x_0, x_1, y, w, c=1, k=0.1):
    if(len(x_0) != len(x_1) != len(y)):
        print("func w() length x and y is not equal!!")
        return 0
    w0_sum = 0.0
    w1_sum = 0.0
    for q in range(0,len(y)):
        w0_sum += x_0[q] * y[q] * (1 - (1 / (1 + m.exp( - y[q] * (w[0] * x_0[q] + w[1] * x_1[q] )))))
        w1_sum += x_1[q] * y[q] * (1 - (1 / (1 + m.exp( - y[q] * (w[0] * x_0[q] + w[1] * x_1[q] )))))
        #sum += x[q-1][n] * y[q-1] * (1 - 1 / (1 - m.exp( - y[q-1] * (w1 * x[q-1][0] + w2 * x[q-1][1] ))))
    
    w_s = [0,0]
    w_s[0] = w[0] + k / len(y) * w0_sum - k * c * w[0]      
    w_s[1] = w[1] + k / len(y) * w1_sum - k * c * w[1]
    return [w_s[0], w_s[1]]

def f(x_0, x_1,y,w):
    if(len(x_0) != len(x_1) != len(y)):
        print("func f() length x and y is not equal!!")
        return 0
    sum = 0.0
    
    for i in range(len(y)):
        sum += m.log(1 + m.exp( - y[i] * (w[0] * x_0[i] + w[1] * x_1[i] )), )
    
    sum /= len(y)
    
    sum += 1/2 * m.sqrt(m.pow(w[0], 2) + m.pow(w[1], 2))
    
    return sum

def train(x_0, x_1, y, c, k):    
    
    prev_w = [0,0]
    new_w = [0,0]
    diff = 1;
    
    for i in range(0,10001):
        
        new_w = w_up(x_0,x_1, y, prev_w, c, k)
        
        if(diff < 1e-5):
            w = new_w
            print("finished")
            print(i)
            break
        else:
            #print(counter)
            diff = np.linalg.norm(np.array(prev_w)-np.array(new_w))
            print(diff)
            prev_w = new_w
            
        #k  = 1 / (counter)
    
    a = 1 / ( 1 + np.exp( - new_w[0] * np.array(data[data.columns[1]]) - new_w[1] * np.array(data[data.columns[2]])))    
    return a




a_0 = train(x_0,x_1, y, 1, k)
a_10 = train(x_0,x_1, y, 10, k)
ans = [str(round(roc_auc_score(y, a_0), 3)), str(round(roc_auc_score(y, a_10), 3))]
print(ans)
