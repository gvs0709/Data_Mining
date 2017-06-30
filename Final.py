import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import time

target=[]
data1=[]
data2=[]

f=open("new_train.csv") 

for line in f:
    #print(line)
    #line = line.strip()
    #print(line)
    vec=line.split(",")
    
    try:
        target_n = int(vec[-1])
        target.append(target_n)
        
    except:
        continue
    
    temp1=[]
    #temp2=[]
    
    for i in xrange(len( vec[:-1])):
        vec[i]=float(vec[i])
        temp1.append(vec[i])
        
  #      try:
  #          vec[i] = float(vec[i])
  #          temp1.append(vec[i])
  #      except ValueError:
  #          temp2.append(vec[i])

    #data2.append(temp2)
    data1.append(temp1)
    
f.close()

g=open("test.csv")

for line in g:
    #print(line)
    #line = line.strip()
    #print(line)
    vec2=line.split(",")

    #temp1=[]
    temp2=[]
    
    for i in xrange(len( vec2[:])):
        vec2[i]=float(vec2[i])
        temp2.append(vec2[i])
        
  #      try:
  #          vec[i] = float(vec[i])
  #          temp1.append(vec[i])
  #      except ValueError:
  #          temp2.append(vec[i])

    data2.append(temp2)
    #data1.append(temp1)
    
g.close()

array2=np.array(data2)
array=np.array(data1)
print np.shape(array)
#print target
print 3*len(array)/100
           
array=array[:, (array != 0).sum(axis=0) >=  2*len(array)/100]
array2=array2[:, np.apply_along_axis(np.count_nonzero, 0, array2) >= 2*len(array2)/100]

print np.shape(array)
print np.shape(array2)
#array=np.array(data2)
#array=np.array(data1)
#print array

'''keys=[]

for i in xrange(len(array.T)):
    conjunto=set(array.T[i])
    pos=list(conjunto)
    keys+=pos

dict_pos={}

for i in xrange(len(keys)):
    dict_pos[keys[i]]=i'''
    

#data2_tratado=[]
#data1_tratado=[]
#for i in xrange(len(data1)):
#for i in xrange(len(data2)):
    #zeros=np.zeros(len(keys))
    
    #for k in data1[i]:
    #for k in data2[i]:
       #zeros[dict_pos[k]]=1           
    #data2_tratado.append(list(zeros))
    #data1_tratado.append(list(zeros))
#print np.array(data2_tratado)
#print np.array(data1_tratado)
#X2 = np.array(data2_tratado)
sc=StandardScaler()
X1=sc.fit_transform(array)
X2=sc.fit_transform(array2)
#X = np.concatenate((X3,X2), axis=1) 
y=np.array(target)
#print X
#print X3

kf=KFold(n_splits=10, shuffle=True)

print "Baseline: ", len(y[y==1])/float(len(y))
acc=[]

for train_index, test_index  in kf.split(X1):
    time1=time.time()
    
    X1_train, X1_test=X1[train_index], X1[test_index]
    y_train, y_test=y[train_index], y[test_index]
    
    ###clf=SVC(kernel="kbf")
    #clf=tree.DecisionTreeClassifier()
    #clf=MLPClassifier(activation='relu', solver='adam', learning_rate='constant', alpha=1e-5, hidden_layer_sizes=(100,), random_state=None)
    #clf=GradientBoostingClassifier()
    clf=XGBClassifier(eta=0.1, max_depth=4, min_child_weight=1, subsample=0.8, colsample_bytree=0.8, gamma=0, objective=binary:logistic, eval_metric=auc)
    clf.fit(X1_train,y_train)
    y_pred=clf.predict(X1_test)
    
    score=accuracy_score(y_pred, y_test)
    #score=roc_auc_score(y_pred, y_test, average='macro')
    acc.append(score)
    print y_pred
    
    print "Demorou: ", time.time() - time1
                                
print "mean accuracy: ", np.mean(acc)

h=open("resposta.txt","r+")
a=clf.predict_proba(X2)

for i in xrange(len(a)):
    h.write(str(a[i,1])+"\n" )

h.truncate()
h.close()
