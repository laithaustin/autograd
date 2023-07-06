import numpy as np
import torch

# Ideas to work on. Expand functionalities for other functions. Abstract away _backward
# member functions so that we can just read from a derivatives.yaml file like pytorch. 
# Then we can expand project to also include creating a neural net from scratch. We
# can use this autograd engine/tensor class to build our network and create a CNN from
# scratch. 

class tensor:
    def __init__(self, data, children=[], requiresGrad=False, op=''):
        self.data = np.array(data) # matrix/vector
        self.requiresGrad = requiresGrad # flag for keeping up with gradient
        self._backward = lambda: None # function for backward pass
        self.grad = 0.0 
        # self.op = op # track which operation came before
        self.children = children # track with tensor is your input

    def __mul__(self, other):
        other = other if isinstance(other, tensor) else tensor(other)
        out = tensor(self.data * other.data, [self,other], True, '*')
        
        # compute gradients for children
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            
        out._backward = _backward
        return out
    
    def dot(self, other):
        other = other if isinstance(other, tensor) else tensor(other)
        out = tensor(np.dot(self.data, other.data), [self, other], True, '@')

        def _backward():
            self.grad += np.dot(out.grad, other.data.T)
            other.grad += np.dot(self.data.T, out.grad)
        
        out._backward = _backward
        return out

    def __add__(self, other):
        other = other if isinstance(other, tensor) else tensor(other)
        out = tensor(self.data + other.data, [self, other], True, '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out
    
    def __pow__(self, val):
        if not isinstance(val, (int, float)):
            return "exponent must be a float or int"
        out = tensor(self.data**val, [self], True, '^')

        def _backward():
            self.grad += out.grad * (val * self.data**(val-1))
        out._backward = _backward
        return out
    
    def relu(self):
        f = lambda x: x if x > 0 else 0
        out = tensor(np.vectorize(f)(self.data), [self], True, 'relu')

        def _backward():
            self.grad += out.grad * (out.data > 0)
        out._backward = _backward
        
        return out
    
    def __truediv__(self, other):
        other = other if isinstance(other, tensor) else tensor(other)
        inv = other**-1
        return self * inv
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __repr__(self):
        return f"MyTensor(data={self.data}, grad={self.grad}, requiresGrad={self.requiresGrad})"

    def log(self):
        out = tensor(np.log(self.data), [self], True, 'log')

        def _backward():
            self.grad += out.grad * (1/self.data)
            print

        out._backward = _backward
        return out

    def sin(self):
        out = tensor(np.log(self.data), [self], True, 'log')
        def _backward():
            self.grad += out.grad * (np.cos(self.data))

        out._backward = _backward
        return out
    
    def cos(self):
        out = tensor(np.log(self.data), [self], True, 'log')
        def _backward():
            self.grad += out.grad * (-np.sin(self.data))

        out._backward = _backward
        return out
    
    def sum(self):
        self.data = np.sum(self.data)
        return self
    
    # recursively call backward - need to build a topological sort
    def backward(self):
        toBuild = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build(child)
                toBuild.append(v) # this is the key to setting the order so that we add bottom layer to top
        
        build(self)
        # print(toBuild)
        self.grad = 1.0
        for node in reversed(toBuild):
            node._backward() # now we can call in topological order

# Test Scripts
# Test 1 will check a simple function that uses add, mult, and log. Verifies results with 
# Pytorch. 
def test1():
    #sample matrices/vectors
    a=[[1.0,2.0], [1.0,2.0]]
    b=[[2,4], [2,4]]
    c=[[1.0,1.0], [1.0,1.0]]
    d=[[2.0,2.0],[2.0,2.0]]

    # sample input
    x, y = tensor(a), tensor(b)
    # weights
    w1 = tensor(c)
    w2 = tensor(d)

    z = x * w1
    g = y * w2
    v = z * g
    output = tensor.log(v)
    output = tensor.sum(output) # gives us a scalar

    # verify with pytorch
    x2,y2 = torch.FloatTensor([[1.0,2.0], [1.0,2.0]]), torch.FloatTensor([[2,4], [2,4]])
    x2.requires_grad = True
    y2.requires_grad = True
    w12 = torch.FloatTensor([[1.0,1.0], [1.0,1.0]])
    w12.requires_grad = True
    w22 = torch.FloatTensor([[2.0,2.0],[2.0,2.0]])
    w22.requires_grad = True

    z2 = x2 * w12
    # z2.requires_grad_()
    g2 = y2 * w22
    # g2.requires_grad_()
    v2 = z2 * g2
    # v2.requires_grad_()
    output2 = torch.log(v2).sum() # should return a scalar

    # backpropogate
    output.backward()

    # naively print all gradients
    print(f"Autograd w1 gradient: {w1.grad}")
    print(f"Autograd w2 gradient: {w2.grad}")

    output2.backward()

    print(f"Pytorch w1 gradient: {w12._grad}")
    print(f"Pytroch w2 gradient: {w22._grad}")

# Test 2 will check for exponents, subtraction, and sin
def test2():
    #sample matrices/vectors
    a=[[1.0,2.0], [1.0,2.0]]
    b=[[2,4], [2,4]]
    c=[[1.0,1.0], [1.0,1.0]]
    d=[[2.0,2.0],[2.0,2.0]]

    # sample input
    x, y = tensor(a) * 10, tensor(b) * 10
    # weights
    w1 = tensor(c) * 10
    w2 = tensor(d) * 10

    z = x / w1
    g = y / w2
    v = z / g
    # l = tensor.log(v)
    bb = tensor.sin(v)
    output = bb**2
    output = tensor.sum(output) # gives us a scalar

    # verify with pytorch
    x2,y2 = torch.FloatTensor([[10.0,20.0], [10.0,20.0]]), torch.FloatTensor([[20,40], [20,40]])
    x2.requires_grad = True
    y2.requires_grad = True
    w12 = torch.FloatTensor([[10.0,10.0], [10.0,10.0]])
    w12.requires_grad = True
    w22 = torch.FloatTensor([[20.0,20.0],[20.0,20.0]])
    w22.requires_grad = True

    z2 = x2 / w12
    # z2.requires_grad_()
    g2 = y2 / w22
    # g2.requires_grad_()
    v2 = z2 / g2
    # v2.requires_grad_()
    # l2 = torch.log(v2)
    bb2 = torch.sin(v2)
    output2 = torch.pow(bb2, 2).sum() # should return a scalar

    # backpropogate
    output.backward()

    # naively print all gradients
    print(f"Autograd w1 gradient: {w1.grad}")
    print(f"Autograd w2 gradient: {w2.grad}")

    output2.backward()

    print(f"Pytorch w1 gradient: {w12._grad}")
    print(f"Pytroch w2 gradient: {w22._grad}")

test2()