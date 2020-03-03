import numpy as np

# 交叉熵损失
class CrossEntropyLoss():
    """
    对最后一层的神经元输出计算交叉熵损失
    """
    def __init__(self):
        self.X = None
        self.labels = None
    
    def __call__(self, X, labels):
        """
        参数：
            X: 模型最后fc层输出
            labels: one hot标注，shape=(batch_size, num_class)
        """
        self.X = X
        self.labels = labels

        return self.forward(self.X)
    
    def forward(self, X):
        """
        计算交叉熵损失
        参数：
            X：最后一层神经元输出，shape=(batch_size, C)
            label：数据onr-hot标注，shape=(batch_size, C)
        return：
            交叉熵loss
        """
        self.softmax_x = self.softmax(X)
        log_softmax = self.log_softmax(self.softmax_x)
        cross_entropy_loss = np.sum(-(self.labels * log_softmax), axis=1).mean()
        return cross_entropy_loss
    
    def backward(self):
        grad_x =  (self.softmax_x - self.labels)  # 返回的梯度需要除以batch_size
        return grad_x / self.X.shape[0]
        
    def log_softmax(self, softmax_x):
        """
        参数:
            softmax_x, 在经过softmax处理过的X
        return: 
            log_softmax处理后的结果shape = (m, C)
        """
        return np.log(softmax_x + 1e-5)
    
    def softmax(self, X):
        """
        根据输入，返回softmax
        代码利用softmax函数的性质: softmax(x) = softmax(x + c)
        """
        batch_size = X.shape[0]
        # axis=1 表示在二维数组中沿着横轴进行取最大值的操作
        max_value = X.max(axis=1)
        #每一行减去自己本行最大的数字,防止取指数后出现inf，性质：softmax(x) = softmax(x + c)
        # 一定要新定义变量，不要用-=，否则会改变输入X。因为在调用计算损失时，多次用到了softmax，input不能改变
        tmp = X - max_value.reshape(batch_size, 1)
        # 对每个数取指数
        exp_input = np.exp(tmp)  # shape=(m, n)
        # 求出每一行的和
        exp_sum = exp_input.sum(axis=1, keepdims=True)  # shape=(m, 1)
        return exp_input / exp_sum