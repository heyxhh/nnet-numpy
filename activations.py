import numpy as np

# 定义Relu层
class Relu(object):
    def __init__(self):
        self.X = None
    
    def __call__(self, X):
        self.X = X
        return self.forward(self.X)
    
    def forward(self, X):
        return np.maximum(0, X)
    
    def backward(self, grad_output):
        """
        grad_output: loss对relu激活输出的梯度
        return: relu对输入input_z的梯度
        """
        grad_relu = self.X > 0  # input_z大于0的提放梯度为1，其它为0
        return grad_relu * grad_output  # numpy中*为点乘


# 定义Tanh层
class Tanh():
    def __init__(self):
        self.X = None
    
    def __call__(self, X):
        self.X = X
        return self.forward(self.X)
    
    def forward(self, X):
        return np.tanh(X)
    
    def backward(self, grad_output):
        grad_tanh = 1 - (np.tanh(self.X)) ** 2
        return grad_output * grad_tanh

# 定义Sigmoid层
class Sigmoid():
    def __init__(self):
        self.X = None
    
    def __call__(self, X):
        self.X = X
        return self.forward(self.X)
    
    def forward(self, X):
        return self._sigmoid(X)
    
    def backward(self, grad_output):
        sigmoid_grad = self._sigmoid(self.X) * (1 - self._sigmoid(self.X))
        return grad_output * sigmoid_grad
    
    def _sigmoid(self, X):
        return 1.0 / (1 + np.exp(-X))