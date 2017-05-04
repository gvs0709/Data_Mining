#------------------------------------------------------#
# Gabriel Villares Silveira  114089936                 #
# Mauricio Miranda           113049797                 #
#------------------------------------------------------#

import numpy as np
from sklearn import svm
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_mldata
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

mnist=fetch_mldata('MNIST original')

X=mnist.data
y=mnist.target

(l, c)=X.shape

amostraTreino=int(l*0.3) #Porcentagem da entrada a ser usada para treino
amostraTeste=int(l*0.7) #Porcentagem da entrada a ser usada para teste

treinoX=np.zeros((amostraTreino, c)) #Matriz de treino, inicializada com zeros
treinoY=np.zeros((amostraTreino, 1)) #Matriz de treino, inicializada com zeros

testeX=np.zeros((amostraTeste, c)) #Matriz de teste, inicializada com zeros
testeY=np.zeros((amostraTeste, 1)) #Matriz de teste, inicializada com zeros

for i in xrange(amostraTreino):
	k=np.random.randint(1, l) #Seleciona linhas aleatorias de X para treino
	treinoX[i]=X[k] #Coloca a linha selecionada na matriz de treino
	treinoY[i]=y[k] #Coloca a classe correspondente da linha escolhida na matriz 

for i in xrange(amostraTeste):
	k=np.random.randint(1, l) #Seleciona linhas aleatorias de X para teste
	testeX[i]=X[k] #Coloca a linha selecionada na matriz de teste

#-------------------------------------Classificadores--------------------------------------#
#clf=MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
#clf=svm.SVC(kernel="linear")
clf=tree.DecisionTreeClassifier()
#------------------------------------------------------------------------------------------#

clf.fit(treinoX, treinoY)
y_pred=clf.predict(X)

print accuracy_score(y, y_pred)
print confusion_matrix(y, y_pred)



'''
separa x e y treino e test
svm
teste de acuracy
matriz de confusao
testar em arvore de decisao, rede neurais, svc ...
'''