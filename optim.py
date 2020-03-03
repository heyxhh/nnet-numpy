class SGD():
    """
    随机梯度下降
    """
    def __init__(self, parameters, lr):
        """
        parameters：模型需要训练的参数
        lr：学习率
        """
        self.parameters = parameters
        self.lr = lr
    
    def update_parameters(self, grads):
        """
        grads: 调用network的backward方法，返回的grads.
        """
        for param, grad in zip(self.parameters, grads):
            param -= self.lr * grad