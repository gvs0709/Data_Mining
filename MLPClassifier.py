#------------------------------------------------------#
# Gabriel Villares Silveira  114089936                 #
# Mauricio Miranda           113049797                 #
#------------------------------------------------------#

import numpy as np
from sklearn import svm
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_mldata

mnist=fetch_mldata('MNIST original')

X=mnist.data
y=mnist.target

(l, c)=X.shape

amostraTreino=int(l*0.3) #Porcentagem da entrada a ser usada para treino
#amostraTeste=int(l*0.7) #Porcentagem da entrada a ser usada para teste

treinoX=np.zeros((amostraTreino, c)) #Matriz de treino, inicializada com zeros
treinoY=np.zeros((amostraTreino, 1)) #Matriz de treino, inicializada com zeros

testeX=np.zeros((2800, c)) #Matriz de teste, inicializada com zeros
testeY=np.zeros((2800, 1)) #Matriz de teste, inicializada com zeros

for i in xrange(amostraTreino):
	k=np.random.randint(1, l) #Seleciona linhas aleatorias de X para treino
	treinoX[i]=X[k] #Coloca a linha selecionada na matriz de treino
	treinoY[i]=y[k] #Coloca a classe correspondente da linha escolhida na matriz 

for i in xrange(2800):
	k=np.random.randint(1, l) #Seleciona linhas aleatorias de X para teste
	testeX[i]=X[k] #Coloca a linha selecionada na matriz de teste

#print testeX
#print X

#--------------------------------------Metodos------------------------------------------#
#clf=MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
#clf=svm.SVC(kernel="linear")
clf=tree.DecisionTreeClassifier()
#----------------------------------------//---------------------------------------------#

clf.fit(treinoX, treinoY)
y_pred=clf.predict(X)

print accuracy_score(y, y_pred)



'''
separa x e y treino e test
svm
teste de acuracy
matriz de confusao
testar em arvore de decisao, rede neurais, svc ...
'''