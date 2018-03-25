import scipy.io
from sklearn import preprocessing
import numpy as np
import math
import random
mat = scipy.io.loadmat('/Users/yzh/Desktop/cour/supervised/hw03_DS5220_Data/data1.mat')
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

def SoftSvmBySmo(reguPara,tol,max_passes,Xtrn,Ytrn):
    dim=Xtrn.shape[1]
    records=Xtrn.shape[0]
    alpha=[0]*records
    b=0
    passes=0
    while passes<max_passes:
      num_changed_alphas=0
      for i in range(0, records):#iterate for the whole alpha list
          yi=np.asscalar(Ytrn[i,:])
          Ei=b
          for k in range(0, records):#calculate Ei
            Ei = Ei+alpha[k] * (np.asscalar(Ytrn[k,:])) * ( np.inner(Xtrn[k,:],Xtrn[i,:]) )
          Ei=Ei-yi
          if (yi*Ei<-tol and alpha[i]<reguPara) or (yi*Ei>tol and alpha[i]>0): #go on, if KKT condition is outside tolerance
            randList=[]
            for pos in range(0,records):
              if pos!=i:
                randList.append(pos)
            j=random.choice(randList)
            Ej=b
            yj=np.asscalar(Ytrn[j,:])
            for k in range(0, records):#calculate Ej
              Ej = Ej + alpha[k] * (np.asscalar(Ytrn[k, :])) * (np.inner(Xtrn[k, :], Xtrn[j, :]))
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
            ijMul=np.inner(Xtrn[i, :], Xtrn[j, :])
            iiMul=np.inner(Xtrn[i, :], Xtrn[i, :])
            jjMul=np.inner(Xtrn[j, :], Xtrn[j, :])
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
                alpha[j]=alpha_jOld
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
      weight=weight+alpha[k]*(np.asscalar(Ytrn[k,:]))*Xtrn[k,:]
    return (weight,b)

para=SoftSvmBySmo(0.1,0.5,30,Xtrn,Ytrn)
print(para)
