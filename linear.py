import numpy as np

# 定义线性层网络
class Linear():
    """
    线性全连接层
    """
    def __init__(self, dim_in, dim_out):
        """
        参数：
            dim_in: 输入维度
            dim_out: 输出维度
        """
        # 初始化参数
        scale = np.sqrt(dim_in / 2)
        self.weight = np.random.standard_normal((dim_in, dim_out)) / scale
        self.bias = np.random.standard_normal(dim_out) / scale
        # self.weight = np.random.randn(dim_in, dim_out)
        # self.bias = np.zeros(dim_out)
        
        self.params = [self.weight, self.bias]
        
    def __call__(self, X):
        """
        参数：
            X：这一层的输入，shape=(batch_size, dim_in)
        return：
            xw + b
        """
        self.X = X
        return self.forward()
    
    def forward(self):
        return np.dot(self.X, self.weight) + self.bias
    
    def backward(self, d_out):
        """
        参数：
            d_out：输出的梯度, shape=(batch_size, dim_out)
        return：
            返回loss对输入 X 的梯度（前一层（l-1）的激活值的梯度）
        """
        # 计算梯度
        # 对input的梯度有batch维度，对参数的梯度对batch维度取平均
        d_x = np.dot(d_out, self.weight.T)  # 输入也即上一层激活值的梯度
        d_w = np.dot(self.X.T, d_out)  # weight的梯度
        d_b = np.mean(d_out, axis=0)  # bias的梯度
        
        return d_x, [d_w, d_b]


# dropout，1D 和 2D feature都可用
class Dropout():
    """
    在训练时随机将部分feature置为0
    """
    def __init__(self, p):
        """
        parameters:
            p: 保留比例
        """
        self.p = p
    
    def __call__(self, X, mode):
        """
        mode: 是在训练阶段还是测试阶段. train 或者 test
        """
        return self.forward(X, mode)
    
    def forward(self, X, mode):
        if mode == 'train':
            self.mask = np.random.binomial(1, self.p, X.shape) / self.p
            out =  self.mask * X
        else:
            out = X
        
        return out
    
    def backward(self, d_out):
        """
        d_out: loss对dropout输出的梯度
        """
        return d_out * self.mask