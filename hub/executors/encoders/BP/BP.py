import numpy as np
from tqdm import trange	


def sigmoid(x): 
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


class BP:	
    def __init__(self, layers):
        self.activation = sigmoid	
        self.activation_deriv = sigmoid_derivative	
        self.weights = []	
        self.bias = []	
        for i in range(1, len(layers)):	
            self.weights.append(np.random.randn(layers[i-1], layers[i]))
            self.bias.append(np.random.randn(layers[i]))

    def train(self, x, y, learning_rate=0.2, epochs=5):	
        x = np.atleast_2d(x)
        n = len(y)	
        p = max(n, epochs)	
        y = np.array(y)
       
        for k in trange(epochs * n):
            if (k+1) % p == 0:
                learning_rate *= 0.5	
            a = [x[k % n]]
            #forward-propagation
            for lay in range(len(self.weights)):
                a.append(self.activation(np.dot(a[lay], self.weights[lay]) + self.bias[lay]))
            #back-propagation
            label = np.zeros(a[-1].shape)
            label[y[k % n]] = 1	#
            error = label - a[-1]	
            deltas = [error * self.activation_deriv(a[-1])]	

            layer_num = len(a) - 2	
            for j in range(layer_num, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[j].T) * self.activation_deriv(a[j]))	
            deltas.reverse()
            for i in range(len(self.weights)):	
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)
                self.bias[i] += learning_rate * deltas[i]

    def predict(self, x):	
        a = np.array(x, dtype=np.float)
        for lay in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[lay]) + self.bias[lay])
        a = list(100 * a/sum(a))	
        i = a.index(max(a))	
        per = []	
        for num in a:
            per.append(str(round(num, 2))+'%')
        return i, per

def test():	
    x = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]]
    y = [0, 1, 2, 3]
    nn = BP([3, 4, 16, 4])
    nn.train(x, y, epochs=5000)
    x2 = [[0, 1, 0], [0, 0, 1]]
    for i in x2:
        print(nn.predict(i))
    

test()