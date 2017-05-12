#------------------------------------------------------#
# Gabriel Villares Silveira	 114089936			       #
# Mauricio Miranda			 113049797	   			   #
#------------------------------------------------------#

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_linear_data(w,b,n):
    dim = len(w)
    y = []
    X = []
    for i in xrange(n):
        x = np.random.uniform(-10,10,dim)
        if(np.dot(w,x) + b > 5):
            y.append(1)
            X.append(x)
        elif(np.dot(w,x) + b < -5):
            y.append(-1)
            X.append(x)
    return np.array(X),y

def plot_plane_and_points(X,y,w,b):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X1 = []
    X2 = []

    for i in xrange(len(y)):
        if (y[i] == -1):
            X1.append(X[i])
        else:
            X2.append(X[i])

    X1 = np.array(X1)
    X2 = np.array(X2)

    ax.scatter(X1[:,0], X1[:,1], X1[:,2], c='blue', marker='^')
    ax.scatter(X2[:,0], X2[:,1], X2[:,2], c='red', marker='o')

    # create x,y
    xx, yy = np.meshgrid(np.arange(-15,15), np.arange(-15,15))

    # calculate corresponding z
    z = (-w[0] * xx - w[1] * yy - b) * 1. /w[2]

    # plot the surface
    ax.plot_surface(xx, yy, z,alpha=0.3, color="green")

data = generate_linear_data(np.array([1,2,3]),0,1000)

w = np.array([ 0.47542968,  0.86705391,  1.43545253] )
b = -6.10673197214e-12

#plot_plane_and_points(data[0], data[1], w, b)
#plt.show()


class PLA: #--Perceptron Learning Algorithm--#
	
	def __init__(self): #Construtor
		self.B=0 #Bias
		self.tol=0.001 #Tolerancia
		self.eta=0.1 #Learning Rate inicial
		self.first=True

	def fit(self, X, y):
		(N, D)=X.shape

		if self.first:
			self.first=False
			self.W=np.zeros(len(X[0]))

		i=0

		while i<N:
			V=np.dot(self.W, X[i])+self.B #Produto Interno

			if np.sign(V)!=np.sign(y[i]):
				self.W+=y[i]*X[i]
				self.B+=y[i]
				i=0

			else:
				i+=1


	def predict(self, X):
		y_=np.dot(X, self.W)

		for i in xrange(len(y_)):
			if y_[i]<0:
				y_[i]=-1

			else:
				y_[i]=1

		return y_

	def score(self, X, y):
		resp=self.predict(X)
		cnt=.0

		for i, r in enumerate(resp):
			if r==y[i]:
				cnt+=1.0

		return cnt/len(y)

	def erro(self, X, y):
		ytgh=np.tanh(np.dot(self.W, X.T)+self.B) #Funcao de Erro
		
		errAnterior=10000 #Valor arbitrario
		errTot=np.linalg.norm((y-ytgh)) #Erro inicial
		varErr=1000 #Chute para variancia do erro inicial

		temp=np.cosh(np.dot(self.W, X.T)+self.B)**(-2) #Secante hiperbolica ao quadrado (derivada de ytgh)

		while varErr>self.tol:
			erros=0
			#varErr=abs(errTot**2-errAnterior)
			#errAnterior=errTot**2

			for i in xrange(len(X)):
				erros+=errTot
				gradErr=-2*np.dot(temp, X)*errTot #Gradiente do erro
				self.W=self.W-self.eta*gradErr #Ajustando W com o learning rate

			#ytgh=np.tanh(np.dot(self.W, X.T)+self.B) #Atualiza a funcao de erro
			#errTot=np.linalg.norm((y-ytgh)) #Atualiza o erro total

			varErr=erros-errAnterior
			errAnterior=erros



		
if __name__ == '__main__':
	test=PLA()
	data = generate_linear_data(np.random.uniform(-1,1,3), np.random.random(), 1000)

	test.fit(data[0], data[1])
	test.erro(data[0], data[1])
	
	plot_plane_and_points(data[0], data[1], test.W, test.B)
	plt.show()

	#print('X='+ str(data[0]))
	#print('W='+ str(test.W))
	#print('b='+ str(test.B))

