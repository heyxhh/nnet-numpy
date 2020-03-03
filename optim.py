import numpy as np

class SGD():
    """
    随机梯度下降
    parameters: 模型需要训练的参数
    lr: float, 学习率
    momentum: float, 动量因子，默认为None不使用动量梯度下降
    """
    def __init__(self, parameters, lr, momentum=None):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum

        if momentum is not None:
            self.velocity = self.velocity_initial()

    def update_parameters(self, grads):
        """
        grads: 调用network的backward方法，返回的grads.
        """
        if self.momentum == None:
            for param, grad in zip(self.parameters, grads):
                param -= self.lr * grad
        else:
            for i in range(len(self.parameters)):
                self.velocity[i] = self.momentum * self.velocity[i] - self.lr * grads[i]
                self.parameters[i] += self.velocity[i]
    
    def velocity_initial(self):
        """
        初始化velocity，按照parameters的参数顺序依次将v初始化为0
        """
        velocity = []
        for param in self.parameters:
            velocity.append(np.zeros_like(param))
        return velocity