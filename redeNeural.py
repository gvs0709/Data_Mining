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

(lin, col)=X.shape
hidden_nodes=np.zeros(col-1) #Array de synapses, contem os pesos

np.random.seed(1) #The seed for the random generator is set so that it will return the same random numbers each time, which is sometimes useful for debugging.

#Now we intialize the weights to random values.
#syn0 are the weights between the input layer and the hidden layer. It is a 3x4 matrix because there are two input weights plus a bias term (=3) and four nodes in the hidden layer (=4).
#syn1 are the weights between the hidden layer and the output layer. It is a 4x1 matrix because there are 4 nodes in the hidden layer and one output.
#Note that there is no bias term feeding the output layer in this example. The weights are initially generated randomly because optimization tends not to work well when all the weights start at the same value.

#synapses
hidden_nodes(0)=2*np.random.random((3,4))-1  # 3x4 matrix of weights ((2 inputs + 1 bias) x 4 nodes in the hidden layer)
hidden_nodes(1)=2*np.random.random((4,1))-1  # 4x1 matrix of weights. (4 nodes x 1 output) - no bias term in the hidden layer.

def nonlin(x, deriv=False): #Definition of the sigmoid function
        return (x*(1-x))
    
    return 1/(1+np.exp(-x))


class RedeNeural:
    def __init__(self, eta, numrounds, hidden_nodes): #Construtor
        self.numrounds=numrounds
        '''
        fit roda até que numrounds
        seja alcançado
        '''
        
        self.eta=eta #learning rate
        self.hidden_nodes=hidden_nodes #Numero de perceptrons (nós) na hidden layer
        
    def fit(self, X, y):
        pass

        
        '''
        o fit deve descobrir a dimensão do problema
        e o número de classes.
        '''
        
    def predict(self, X):
        pass
        #predict deve retornar as classes para X