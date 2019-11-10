# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

start=time.process_time()
#pre-process data
path = r'C:\Users\Oval Liu\Desktop\iris.data'
iris=pd.read_csv(path)
iris.columns=['sl','sw','pl','pw','type_raw']
X=iris.iloc[0:,1:3]

Y=iris.type_raw.astype('category')
y1=Y.cat.rename_categories({'Iris-setosa':-1,'Iris-versicolor':1,'Iris-virginica':0}) #.cat is attritbute associated with categorical
x=np.c_[X.iloc[0:, 0], X.iloc[0:, 1]] #Arraying
y2=np.asarray(y1)
y =np.where(y2>=0,1,-1)

#function settings
def perceptron(x,y,max_iter=200,lr=0.01):#'max_iter'is the max iteration, 'lr' is the learning rate 
    w = np.zeros(x.shape[1]) #initialize w and b
    b = 0

    for i in range(max_iter):
        y_pre = predict(w,x,b)
        error = y*y_pre # error = 1 or -1
        index = np.argmin(error) #return the index of error's min

        delta = lr*y[index]
        w += delta*x[index]
        b += delta
    return w,b

def predict(w,x,b):
    rs=np.asarray(x,dtype=np.float32).dot(w)+b
    return np.sign(rs).astype(np.float32)

#processing
w,b = perceptron(x,y)
z = predict(w,x,b)
accuracy = sum([z == y] ).mean()
print(accuracy)

end=time.process_time()
print('Processing time:',end-start)
#visualization
m = np.linspace(min(X.iloc[0:,0]),max(X.iloc[0:,0]),100)
n = -w[0]*m/w[1]-b/w[1]
plt.scatter(X.iloc[0:,0], X.iloc[0:,1], c=y, cmap=plt.cm.Paired)
plt.plot(m,n)
plt.show()
