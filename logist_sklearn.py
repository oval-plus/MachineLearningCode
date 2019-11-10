# -*- coding: utf-8 -*-

import time
from sklearn import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

start=time.process_time()
X, y = datasets.load_iris(return_X_y=True)

p=0.3
def logistic(p):
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=p)
    Clif = LogisticRegression(solver='newton-cg',random_state=0,multi_class='multinomial')
    Clif.fit(x_train,y_train)
    KK=Clif.predict(x_test)
    error=np.mean((abs(KK-y_test)))
    return(error)

error_list=[]
for i in range(0,1000):
    error_list.append(logistic(p))
s=np.mean(error_list)
print(s)

end=time.process_time()
print('Processing time:',end-start)