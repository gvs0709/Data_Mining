import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

target=[]
data1=[] 
data2=[]

f=open("crx.data")

for line in f:
    vec=line.split(",")
    target.append(int(vec[-1]))
    temp1=[]
    temp2=[]

    for i in xrange(len( vec[:-1])):
        try:
            vec[i]=float(vec[i])
            temp1.append(vec[i])

        except ValueError:
            temp2.append(vec[i])

    data2.append(temp2)
    data1.append(temp1)

array=np.array(data2)

#print np.array(data1) #Colunas Numericas

#array = np.array(data2)
#print array #Colunas Categoricas

keys = []

for i in xrange(len(array.T)):
    conjunto=set(array.T[i])
    pos=list(conjunto)
    keys+=pos

dict_pos={}

for i in xrange(len(keys)):
    dict_pos[keys[i]]=i

data2_tratado=[]

for i in xrange(len(data2)):
    zeros=np.zeros(len(keys))

    for k in data2[i]:
        zeros[dict_pos[k]]=1

    data2_tratado.append(list(zeros))

#print np.array(data2_tratado)

#Sem StandardScaler
'''
X2=np.array(data2_tratado)
X3=np.array(data1)
X=np.concatenate((X3,X2), axis=1)

y=np.array(target)

print X

kf=KFold(n_splits=10, shuffle=True)

print "Baseline: ", len(y[y==1])/float(len(y))

acc=[]

for train_index, test_index in kf.split(X):
    X_train, X_test=X[train_index], X[test_index]
    y_train, y_test=y[train_index], y[test_index]

    clf=SVC(kernel="rbf")
    clf.fit(X_train, y_train)

    y_pred=clf.predict(X_test)
    score=accuracy_score(y_pred, y_test)
    acc.append(score)

print "mean accuracy: ", np.mean(acc)
'''

#Utilizando o StandardScaler
X2=np.array(data2_tratado)

sc=StandardScaler()

X3=sc.fit_transform(data1)
X=np.concatenate((X3,X2), axis=1)

y=np.array(target)

print X

kf=KFold(n_splits=10, shuffle=True)

print "Baseline: ", len(y[y==1])/float(len(y))

acc=[]

for train_index, test_index in kf.split(X):
    X_train, X_test=X[train_index], X[test_index]
    y_train, y_test=y[train_index], y[test_index]

    clf=SVC(kernel = "rbf")
    clf.fit(X_train,y_train)

    y_pred=clf.predict(X_test)
    score=accuracy_score(y_pred, y_test)
    acc.append(score)
    
print "mean accuracy: ", np.mean(acc)