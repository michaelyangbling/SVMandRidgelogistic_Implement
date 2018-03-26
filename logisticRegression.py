#Ridge Logistic Regression
#training data in dataset2 are somewhat mixed with each other,
#which make them not so separable as dataset1
#thus the trained model is not as good as dataset1
import scipy.io
from sklearn import preprocessing
import numpy as np
import math
mat = scipy.io.loadmat('/Users/yzh/Desktop/cour/supervised/hw03_DS5220_Data/data2.mat')
print("class is balanced since class-1 occupies proportion of: "+str(mat["Y_trn"].mean()))
scaler = preprocessing.StandardScaler().fit(mat["X_trn"]) #feature scaling
Xtrn   = scaler.transform(mat["X_trn"])
Xtst=scaler.transform(mat["X_tst"])

Xtrn=np.append(Xtrn,np.ones((Xtrn.shape[0],1)),axis=1) # add ones
Xtst=np.append(Xtst,np.ones((Xtst.shape[0],1)),axis=1)

def ridgeLogis(Xtrn, Ytrn, learnRate,reguPara,stopWhen):
    featureNum=Xtrn.shape[1]
    weight=np.zeros((featureNum,1))
    while True:
      vecSum=np.zeros((featureNum,1))
      for row in range(0,Xtrn.shape[0]):
        feature=Xtrn[row,:]
        feature.shape=(featureNum,1)
        vecSum=vecSum-feature*(np.asscalar(Ytrn[row,0]) - \
        1/(1+math.exp(-np.asscalar(np.matmul(weight.transpose(),
                                             feature)))))
      vecSum=vecSum+2*reguPara*weight
      if np.asscalar(np.matmul(vecSum.transpose(),vecSum))<=stopWhen: #10**(-3)
          return weight+(-learnRate)*vecSum
      weight=weight+(-learnRate)*vecSum

weight=ridgeLogis(Xtrn,mat["Y_trn"],0.01,0,10**(-2))
print("weight:")
print(weight)

def predict(Xtst,weight):
  result=[]
  for row in range(0,Xtst.shape[0]):
    feature=Xtst[row,:]
    feature.shape=(1,Xtst.shape[1])
    result.append(1/(1 +math.exp(-np.asscalar(np.matmul(feature,weight))) ))
  return result


result=np.array(predict(Xtst,weight))
trainScore=np.array(predict(Xtrn,weight))
from sklearn.metrics import roc_curve, auc #use Area under roc-curve Metric
a,b,c=roc_curve(mat["Y_tst"].flatten(),result)
print("test error: Area under roc-curve is "+str(auc(a,b)))
a,b,c=roc_curve(mat["Y_trn"].flatten(),trainScore)
print("training error: Area under roc-curve is "+str(auc(a,b)))

if np.asscalar(weight[0])!=0: # x1 as a function of x2
        slope=-np.asscalar(weight[1])/np.asscalar(weight[0])
        intercept=-np.asscalar(weight[2])/np.asscalar(weight[0])

import matplotlib.pyplot as plt
plt.plot([-3,3],np.array([-3,3])*slope+intercept)#order of appearance
trnX1=Xtrn[:,0]
trnX2=Xtrn[:,1]
trnClass=mat["Y_trn"].flatten().tolist()
for i in range(0,len(trnClass)): #0 corresponds to color green,1:red
    if trnClass[i]==0:
        trnClass[i]="green"
    else:
        trnClass[i] = "red"
plt.scatter(trnX2,trnX1,c=trnClass)
print("training-set plot")
plt.show()

plt.plot([-3,3],np.array([-3,3])*slope+intercept)
tstX1=mat["X_tst"][:,0]
tstX2=mat["X_tst"][:,1]
tstClass=mat["Y_tst"].flatten().tolist()
for i in range(0,len(tstClass)): #0 corresponds to color green,1:red
    if tstClass[i]==0:
        tstClass[i]="green"
    else:
        tstClass[i] = "red"
plt.scatter(tstX2,tstX1,c=tstClass)
print("test-set plot")
plt.show()
