# -*- coding: utf-8 -*-

import time
from sklearn import *
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
import numpy as np

start=time.process_time()
iris=datasets.load_iris()
X = iris.data[ : , 1:3]  
y = iris.target
y[y==2]=1
y[y==0]=-1
x = np.c_[X[:, 0], X[:, 1]]

p=0.3
def perceptron(p):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=p)
    Clif = Perceptron(random_state=0)
    Clif.fit(x_train,y_train)
    KK=Clif.predict(x_test)
    error=np.mean((abs(KK-y_test)))
    return(error)

error_list=[]
for i in range(0,1000):
    error_list.append(perceptron(p))
s=np.mean(error_list)
print(s)

end=time.process_time()
print('Processing time:',end-start)