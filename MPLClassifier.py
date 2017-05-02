import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.datasets import fetch_mldata

mnist=fetch_mldata('MNIST original')

X=mnist.data
y=mnist.target


#clf=MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf=svm.SVC(kernel="rbf")

#clf.fit(X, y)                         


'''
separa x e y treino e test
svm
teste de acuracy
matriz de confusao
testar em arvore de decisao, rede neurais, svc ...
'''