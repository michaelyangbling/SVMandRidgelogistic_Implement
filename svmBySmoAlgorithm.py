import scipy.io
from sklearn import preprocessing
import numpy as np
import math
import random
mat = scipy.io.loadmat('/Users/yzh/Desktop/cour/supervised/hw03_DS5220_Data/data2.mat')
print("class is balanced since class-1 occupies proportion of: "+str(mat["Y_trn"].mean()))
scaler = preprocessing.StandardScaler().fit(mat["X_trn"]) #feature scaling
Xtrn   = scaler.transform(mat["X_trn"])
Xtst=scaler.transform(mat["X_tst"])

def convert(x):
    if x==1:
        return 1
    else:
        return -1
vfunc=np.vectorize(convert)
Ytrn=vfunc(mat["Y_trn"])

# Xtrn=np.append(Xtrn,np.ones((Xtrn.shape[0],1)),axis=1) # add ones
# Xtst=np.append(Xtst,np.ones((Xtst.shape[0],1)),axis=1)

def SoftSvmBySmo(reguPara,tol,max_passes,data,label):
    dim=data.shape[1]
    records=data.shape[0]
    alpha=[0]*records
    b=0
    passes=0
    while passes<max_passes:
      num_changed_alphas=0
      for i in range(0, records):#iterate for the whole alpha list
          yi=np.asscalar(label[i,:])
          Ei=b
          for k in range(0, records):#calculate Ei
            Ei = Ei+alpha[k] * (np.asscalar(label[k,:])) * ( np.inner(data[k,:],data[i,:]) )
          Ei=Ei-yi
          if (yi*Ei<-tol and alpha[i]<reguPara) or (yi*Ei>tol and alpha[i]>0): #go on, if KKT condition is outside tolerance
            randList=[]
            for pos in range(0,records):
              if pos!=i:
                randList.append(pos)
            j=random.choice(randList)
            Ej=b
            yj=np.asscalar(label[j,:])
            for k in range(0, records):#calculate Ej
              Ej = Ej + alpha[k] * (np.asscalar(label[k, :])) * (np.inner(data[k, :], data[j, :]))
            Ej = Ej - yj
            alpha_iOld=alpha[i]
            alpha_jOld=alpha[j]
            # if yi!=yj:
            #     print("a")
            if yi!=yj:
                low=max(0,alpha[j]-alpha[i])
                high=min(reguPara,reguPara+alpha[j]-alpha[i])
            else:
                low = max(0, alpha[i] +alpha[j]-reguPara)
                high=min(reguPara,alpha[i] +alpha[j])
            if low==high:
                continue
            ijMul=np.inner(data[i, :], data[j, :])
            iiMul=np.inner(data[i, :], data[i, :])
            jjMul=np.inner(data[j, :], data[j, :])
            eta=2*ijMul-iiMul-jjMul
            if eta>=0:
                continue
            # if yi!=yj:
            #     print("x")
            alpha[j]=alpha[j]-yj*(Ei-Ej)/eta
            if alpha[j]>high:
                alpha[j]=high
            elif alpha[j]<low:
                alpha[j] = low
            if abs(alpha[j]-alpha_jOld)<10**(-5):
                #alpha[j]=alpha_jOld
                continue
            alpha[i]=alpha_iOld+yi*yj*(alpha_jOld-alpha[j]) #update alpha[i]
            b1 = b - Ei - yi * (alpha[i] - alpha_iOld) * iiMul - yj * (alpha[j] - alpha_jOld) * ijMul
            b2 = b - Ej - yi * (alpha[i] - alpha_iOld) * ijMul - yj * (alpha[j] - alpha_jOld) * jjMul
            if alpha[i]>0 and alpha[i]<reguPara:
                b=b1
            elif alpha[j]>0 and alpha[j]<reguPara:
                b=b2
            else:
                b=(b1+b2)/2
            num_changed_alphas+=1
      if num_changed_alphas==0:
          passes+=1
      else:
          passes=0
    weight=np.array([0]*dim)
    for k in range(0,records):
      weight=weight+alpha[k]*(np.asscalar(label[k,:]))*data[k,:]
    return (weight,b)

# para=SoftSvmBySmo(0.1,0.5,30,Xtrn,Ytrn)
# print(para)

def predict(test,param):
    prediction=[]
    for i in range(0, test.shape[0]):
        if np.inner(param[0],test[i,:])+param[1]>=0:
            label=1
        else:
            label=0
        prediction.append(label)
    return prediction

def get_accuracy(prediction, realValue): #prediction, realValue as lists
    numTrue=0
    for i in range(0,len(prediction)):
        if prediction[i]==realValue[i]:
            numTrue+=1
    return numTrue/len(prediction)

trnAcc=[]
testAcc=[]
for i in list(map(lambda x: x/25, range(1,250,20))):
    para=SoftSvmBySmo(i,0.001,5,Xtrn,Ytrn)
    print("a training round finished")
    trnAcc.append( get_accuracy( predict(Xtrn,para), mat["Y_trn"].flatten().tolist() ) )
    testAcc.append( get_accuracy( predict(Xtst,para), mat["Y_tst"].flatten().tolist() ) )
import matplotlib.pyplot as plt
print("trainingErr dependent on reguPara C, green")
plt.scatter(list(map(lambda x: x/25, range(1,250,20))), trnAcc,c='green')
#plt.show()

plt.scatter(list(map(lambda x: x/25, range(1,250,20))), testAcc,c='red')
print("testErr dependent on reguPara C, red")
plt.show()

print("comparing this SMO algorithm with sklearn-SVC")
from sklearn import svm
trnAcc=[]
testAcc=[]

for i in list(map(lambda x: x/25, range(1,2500,10))):
    clf=svm.SVC(C=i,kernel='linear',tol=0.001)
    clf.fit(Xtrn,mat['Y_trn'].flatten())
    print("a training round finished")
    trnAcc.append( get_accuracy( clf.predict(Xtrn).tolist(), mat["Y_trn"].flatten().tolist()))
    testAcc.append(get_accuracy(clf.predict(Xtst).tolist(), mat["Y_tst"].flatten().tolist()))
print("trainingErr dependent on reguPara C,green")
plt.scatter(list(map(lambda x: x/25, range(1,2500,10))), trnAcc,c='green')
#plt.show()

plt.scatter(list(map(lambda x: x/25, range(1,2500,10))), testAcc,c='red')
print("testErr dependent on reguPara C,red")
plt.show()
