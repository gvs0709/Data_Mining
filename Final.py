import time
import numpy as np
#import pandas as pd
import xgboost as xgb
import numpy.matlib
from xgboost.sklearn import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, normalize
#import matplotlib.pylab as plt
#%matplotlib inline
#from matplotlib.pylab import rcParams
#rcParams['figure.figsize'] = 12, 4

#train=pd.read_csv('new_train.csv')
#target='TARGET'
#IDcol='ID'

target=[]
data1=[]
data2=[]

f=open("new_train.csv") 

for line in f:
    vec=line.split(",")
    
    try:
        target_n = int(vec[-1])
        target.append(target_n)
        
    except:
        continue
    
    temp1=[]
    
    for i in xrange(len( vec[:-1])):
        vec[i]=float(vec[i])
        temp1.append(vec[i])

    data1.append(temp1)
    
f.close()

g=open("test.csv")

for line in g:
    vec2=line.split(",")
    temp2=[]
    
    for i in xrange(len( vec2[:])):
        vec2[i]=float(vec2[i])
        temp2.append(vec2[i])

    data2.append(temp2)
    
g.close()

array2=np.array(data2)
array=np.array(data1)
print np.shape(array)

'''a=np.matlib.zeros((np.shape(array)[0], np.shape(array)[1]+1))
a2=np.matlib.zeros((np.shape(array2)[0], np.shape(array2)[1]+1))

for i in xrange(len(array)):
    k=0

    for j in xrange(np.shape(array)[1]):
        if array[i, j]==0:
            k=k+1

    a[i, :-1]=array[i, :]
    a[i, -1]=k

for i in xrange(len(array2)):
    k=0

    for j in xrange(np.shape(array2)[1]):
        if array2[i, j]==0:
            k=k+1

    a2[i, :-1]=array2[i, :]
    a2[i, -1]=k

array=a
array2=a2'''

array=array[:, np.apply_along_axis(np.count_nonzero, 0, array) >= 14*len(array)/100]
array2=array2[:, np.apply_along_axis(np.count_nonzero, 0, array2) >= 14*len(array2)/100]

print np.shape(array)
print np.shape(array2)

array=normalize(array, norm='l2', axis=0, copy='False')
array2=normalize(array2, norm='l2', axis=0, copy='False')

sc=StandardScaler()
X1=sc.fit_transform(array)
X2=sc.fit_transform(array2)
y=np.array(target)

kf=KFold(n_splits=10, shuffle=True)

print "Baseline: ", len(y[y==1])/float(len(y))
acc=[]

#melhor=[0, 0]

#for i in xrange(0, 3, 1):

for train_index, test_index  in kf.split(X1):
    time1=time.time()
    
    X1_train, X1_test=X1[train_index], X1[test_index]
    y_train, y_test=y[train_index], y[test_index]
    
    #clf=MLPClassifier(activation='relu', solver='adam', learning_rate='constant', alpha=1e-5, hidden_layer_sizes=(100,), random_state=None)
    clf=XGBClassifier(learning_rate=0.1, max_depth=3, min_child_weight=5, subsample=0.8, colsample_bytree=0.6, gamma=0.1, scale_pos_weight=1.3, objective='binary:logistic', eval_metric='auc', n_jobs=2, random_state=27)
    clf.fit(X1_train,y_train)
    y_pred=clf.predict_proba(X1_test)
    
    #score=accuracy_score(y_pred, y_test)
    score=roc_auc_score(y_test, y_pred[:, 1], average='weighted')
    acc.append(score)
    #print y_pred
    
    print "Demorou: ", time.time() - time1
                                
print "Mean roc_auc: ", np.mean(acc)

    #if np.mean(acc)>melhor[1]:
    #    melhor=[i, np.mean(acc)]

h=open("resposta.txt","r+")
a=clf.predict_proba(X2)

for i in xrange(len(a)):
    h.write(str(a[i,1])+"\n" )

h.truncate()
h.close()

#print "melhor=", melhor

'''def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['TARGET'],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['TARGET'].values, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['TARGET'], dtrain_predprob)
                    
    #feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    #feat_imp.plot(kind='bar', title='Feature Importances')
    #plt.ylabel('Feature Importance Score')

#Choose all predictors except target & IDcols
predictors = [x for x in train.columns if x not in [target]]
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 n_jobs=4,
 scale_pos_weight=1,
 random_state=27)
modelfit(xgb1, train, predictors)

param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', n_jobs=4, scale_pos_weight=1, random_state=27), 
 param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(train[predictors],train[target])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

param_test2 = {
 'max_depth':[4,5,6],
 'min_child_weight':[4,5,6]
}
gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=5,
 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', n_jobs=4, scale_pos_weight=1,random_state=27), 
 param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch2.fit(train[predictors],train[target])
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_

param_test2b = {
 'min_child_weight':[6,8,10,12]
}
gsearch2b = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=4,
 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', n_jobs=4, scale_pos_weight=1,random_state=27), 
 param_grid = param_test2b, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch2b.fit(train[predictors],train[target])

modelfit(gsearch3.best_estimator_, train, predictors)
gsearch2b.grid_scores_, gsearch2b.best_params_, gsearch2b.best_score_

param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', n_jobs=4, scale_pos_weight=1,random_state=27), 
 param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch3.fit(train[predictors],train[target])
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

xgb2 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=4,
 min_child_weight=6,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 n_jobs=4,
 scale_pos_weight=1,
 random_state=27)
modelfit(xgb2, train, predictors)

param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', n_jobs=4, scale_pos_weight=1,random_state=27), 
 param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch4.fit(train[predictors],train[target])
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

param_test5 = {
 'subsample':[i/100.0 for i in range(75,90,5)],
 'colsample_bytree':[i/100.0 for i in range(75,90,5)]
}
gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', n_jobs=4, scale_pos_weight=1,random_state=27), 
 param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch5.fit(train[predictors],train[target])

param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
 min_child_weight=6, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', n_jobs=4, scale_pos_weight=1,random_state=27), 
 param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch6.fit(train[predictors],train[target])
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_

param_test7 = {
 'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
}
gsearch7 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
 min_child_weight=6, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', n_jobs=4, scale_pos_weight=1,random_state=27), 
 param_grid = param_test7, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch7.fit(train[predictors],train[target])
gsearch7.grid_scores_, gsearch7.best_params_, gsearch7.best_score_

xgb3 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=4,
 min_child_weight=6,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'binary:logistic',
 n_jobs=4,
 scale_pos_weight=1,
 random_state=27)
modelfit(xgb3, train, predictors)

xgb4 = XGBClassifier(
 learning_rate =0.01,
 n_estimators=5000,
 max_depth=4,
 min_child_weight=6,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'binary:logistic',
 n_jobs=4,
 scale_pos_weight=1,
 random_state=27)
modelfit(xgb4, train, predictors)'''
