import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

target=[]
data1=[]
data2=[]

f=open("test.csv")
#f=open("new_train.csv") 


for line in f:
    vec=line.split(",")
    target.append(float(vec[-1]))
    temp1=[]
    #temp2=[]
    
    for i in xrange(len( vec[:-1])):
        vec[i]=float(vec[i])
        temp1.append(vec[i])
        
        '''try:
            vec[i] = float(vec[i])
            temp1.append(vec[i])
        except ValueError:
            temp2.append(vec[i])'''

    #data2.append(temp2)
    data1.append(temp1)

#array=np.array(data2)
array=np.array(data1)
print np.array(data1)
#array=np.array(data2)
#array=np.array(data1)
#print array

keys=[]

for i in xrange(len(array.T)):
    conjunto=set(array.T[i])
    pos=list(conjunto)
    keys+=pos

dict_pos={}

for i in xrange(len(keys)):
    dict_pos[keys[i]]=i

#data2_tratado=[]
data1_tratado=[]

for i in xrange(len(data1)):
#for i in xrange(len(data2)):
    zeros=np.zeros(len(keys))
    
    for k in data1[i]:
    #for k in data2[i]:
        zeros[dict_pos[k]]=1
             
    #data2_tratado.append(list(zeros))
    data1_tratado.append(list(zeros))
    
#print np.array(data2_tratado)
print np.array(data1_tratado)

#X2 = np.array(data2_tratado)
sc=StandardScaler()

X3=sc.fit_transform(data1)
#X = np.concatenate((X3,X2), axis=1) 
y=np.array(target)

#print X
print X3

kf=KFold(n_splits=10, shuffle=True)

print "Baseline: ", len(y[y==1])/float(len(y))

acc=[]

for train_index, test_index in kf.split(X3):
#for train_index, test_index in kf.split(X):
    #X_train, X_test=X[train_index], X[test_index]
    X3_train, X3_test=X3[train_index], X3[test_index]
    y_train, y_test=y[train_index], y[test_index]
    
    #clf=SVC(kernel="rbf")
    clf=tree.DecisionTreeClassifier()
    
    #clf.fit(X_train,y_train)
    clf.fit(X3_train,y_train)
    
    #y_pred=clf.predict(X_test)
    y_pred=clf.predict(X3_test)
    score=accuracy_score(y_pred, y_test)
    
    acc.append(score)
    
print "mean accuracy: ", np.mean(acc)