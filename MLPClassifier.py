#------------------------------------------------------#
# Gabriel Villares Silveira  114089936                 #
# Mauricio Miranda           113049797                 #
#------------------------------------------------------#

import numpy as np
from sklearn import svm
from sklearn import tree
from sklearn.neural_network import MLPClassifier
#from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_mldata
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

mnist=fetch_mldata('MNIST original')

X=mnist.data
y=mnist.target

#sc=StandardScaler
#X2=sc.fit_transform(X)

(l, c)=X.shape
#(l, c)=X2.shape

amostraTreino=int(l*0.3) #Porcentagem da entrada (X) a ser usada para treino
amostraTeste=int(l*0.4) #Porcentagem da entrada (X) a ser usada para teste

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
	testeY[i]=y[k] #Coloca a classe correspondente da linha escolhida na matriz

#-------------------------------------Classificadores--------------------------------------#
clf=MLPClassifier(activation='identity', solver='adam', learning_rate='constant', alpha=1e-5, hidden_layer_sizes=(5, 5), random_state=1) #Acuracia em torno de 86,4% a 88,6%

#clf=svm.SVC(kernel="linear") #Acuracia em torno de 92% a 93,4%

#clf=tree.DecisionTreeClassifier() #Acuracia em torno de 87,1% a 87,5%
#------------------------------------------------------------------------------------------#

clf.fit(treinoX, treinoY)
y_pred=clf.predict(testeX)

print accuracy_score(testeY, y_pred)
print confusion_matrix(testeY, y_pred)

'''
Notas: 1) No MLPClassifier trocar a funcao de ativacao da padrao (relu) para identity (f(x)=x), aumentou a acuracia de 10~11% para algo perto de 60%,
alem disso, aumentar o numero de hidden_layer_size=(5, 2) para hidden_layer_size=(5, 5) junto com o uso do solver padrao 'adam' fez a acuracia alcancar os niveis descritos acima

	   2) No svm o kernel='linear' obteve a maior acuracia, que eh a descrita acima
'''