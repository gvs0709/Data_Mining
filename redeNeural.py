#------------------------------------------------------#
# Gabriel Villares Silveira  114089936                 #
# Mauricio Miranda           113049797                 #
#------------------------------------------------------#

import numpy as np

#input data
X=np.array([[0,0,1],  #Note: The third column is for accommodating the bias term and is not part of the input. 
            [0,1,1],
            [1,0,1],
            [1,1,1]])

# The output of the exclusive OR function follows. 

#output data
y=np.array([[0],
             [1],
             [1],
             [0]])

(linx, colx)=X.shape
(liny, coly)=y.shape


np.random.seed(1) #The seed for the random generator is set so that it will return the same random numbers each time, which is sometimes useful for debugging.

#Now we intialize the weights to random values.
#syn0 are the weights between the input layer and the hidden layer. It is a 3x4 matrix because there are two input weights plus a bias term (=3) and four nodes in the hidden layer (=4).
#syn1 are the weights between the hidden layer and the output layer. It is a 4x1 matrix because there are 4 nodes in the hidden layer and one output.
#Note that there is no bias term feeding the output layer in this example. The weights are initially generated randomly because optimization tends not to work well when all the weights start at the same value.

#synapses
syn0=2*np.random.random((colx,linx))-1  # 3x4 matrix of weights ((2 inputs + 1 bias) x 4 nodes in the hidden layer)
syn1=2*np.random.random((liny,coly))-1  # 4x1 matrix of weights. (4 nodes x 1 output) - no bias term in the hidden layer.

(lin, col)=syn0.shape

def nonlin(x, deriv=False): #Definition of the sigmoid function
    if(deriv==True):
        return (x*(1-x))
    
    return 1/(1+np.exp(-x))


class RedeNeural:
    def __init__(self, eta=0.1,numrounds=60000, hidden_nodes=col): #Construtor
        self.numrounds=numrounds #fit roda ate que numrounds seja alcancado
        self.eta=eta #learning rate inicial
        self.hidden_nodes=hidden_nodes #Numero de perceptrons (nos) na hidden layer
        
    def fit(self, X, y): #This is the main training loop. The output shows the evolution of the error between the model and desired. The error steadily decreases.
        global syn0
        global syn1
        global l2
        
        for j in xrange(self.numrounds): #training step
            #Calculate forward through the network.
            l0=X
            l1=nonlin(np.dot(l0, syn0))
            l2=nonlin(np.dot(l1, syn1))
             
            l2_error=y - l2 #Back propagation of errors using the chain rule.

            if(j%10000)==0: #Only print the error every 10000 steps, to save time and limit the amount of output. 
                print("Error: " + str(np.mean(np.abs(l2_error))))        
        
            l2_delta=l2_error*nonlin(l2, deriv=True)
            l1_error=l2_delta.dot(syn1.T)
            l1_delta=l1_error * nonlin(l1,deriv=True)
            
            #update weights (no learning rate term)
            syn1+=l1.T.dot(l2_delta)#*self.eta
            syn0+=l0.T.dot(l1_delta)#*self.eta
        
    def predict(self, X): #predict deve retornar as classes para X
        y_=l2

        for i in xrange(len(y_)):
            if y_[i]<0.7: #saber qual a condicao de separacao
                y_[i]=0 #classe 1

            else:
                y_[i]=1 #classe 2

        print 'y_=', y_
        return y_


if __name__ == '__main__':
    test=RedeNeural()
    test.fit(X, y)
    test.predict(X)