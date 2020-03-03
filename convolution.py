import numpy as np

# 定义conv2d
class Conv2d():
    def __init__(self, in_channels, n_filter, filter_size, padding, stride):
        """
        parameters:
            in_channel: 输入feature的通道数
            n_filter: 卷积核数目
            filter_size: 卷积核的尺寸(h_filter, w_filter)
            padding: 0填充数目
            stride: 卷积核滑动步幅
        """
        self.in_channels = in_channels
        self.n_filter = n_filter
        self.h_filter, self.w_filter = filter_size
        self.padding = padding
        self.stride = stride
        
        # 初始化参数,卷积网络的参数size与输入的size无关
        self.W = np.random.randn(n_filter, self.in_channels, self.h_filter, self.w_filter) / np.sqrt(n_filter / 2.)
        self.b = np.zeros((n_filter, 1))
        
        self.params = [self.W, self.b]
        
    def __call__(self, X):
        # 计算输出feature的尺寸
        self.n_x, _, self.h_x, self.w_x = X.shape
        self.h_out = (self.h_x + 2 * self.padding - self.h_filter) / self.stride + 1
        self.w_out = (self.w_x + 2 * self.padding - self.w_filter) / self.stride + 1
        if not self.h_out.is_integer() or not self.w_out.is_integer():
            raise Exception("Invalid dimensions!")
        self.h_out, self.w_out = int(self.h_out), int(self.w_out)
        
        # 声明Img2colIndices实例
        self.img2col_indices = Img2colIndices((self.h_filter, self.w_filter), self.padding, self.stride)
        
        return self.forward(X)
    
    def forward(self, X):
        # 将X转换成col
        self.x_col = self.img2col_indices.img2col(X)
        
        # 转换参数W的形状，使它适合与col形态的x做计算
        self.w_row = self.W.reshape(self.n_filter, -1)
        
        # 计算前向传播
        out = self.w_row @ self.x_col + self.b  # @在numpy中相当于矩阵乘法，等价于numpy.matmul()
        out = out.reshape(self.n_filter, self.h_out, self.w_out, self.n_x)
        out = out.transpose(3, 0, 1, 2)
        
        return out
    
    def backward(self, d_out):
        """
        parameters:
            d_out: loss对卷积输出的梯度
        """
        # 转换d_out的形状
        d_out_col = d_out.transpose(1, 2, 3, 0)
        d_out_col = d_out_col.reshape(self.n_filter, -1)
        
        d_w = d_out_col @ self.x_col.T
        d_w = d_w.reshape(self.W.shape)  # shape=(n_filter, d_x, h_filter, w_filter)
        d_b = d_out_col.sum(axis=1).reshape(self.n_filter, 1)
        
        d_x = self.w_row.T @ d_out_col
        # 将col态的d_x转换成image格式
        d_x = self.img2col_indices.col2img(d_x)
        
        return d_x, [d_w, d_b]


# 定义maxpool
class Maxpool():
    def __init__(self, size, stride):
        """
        parameters:
            size: maxpool框框的尺寸,int类型
            stride: maxpool框框的滑动步幅，一般设计步幅和size一样
        """
        self.size = size  # maxpool框的尺寸
        self.stride = stride
        
    def __call__(self, X):
        """
        parameters:
            X: 输入feature，shape=(batch_size, channels, height, width)
        """
        self.n_x, self.c_x, self.h_x, self.w_x = X.shape
        # 计算maxpool输出尺寸
        self.h_out = (self.h_x - self.size) / self.stride + 1
        self.w_out = (self.w_x - self.size) / self.stride + 1
        if not self.h_out.is_integer() or not self.w_out.is_integer():
            raise Exception("Invalid dimensions!")
        self.h_out, self.w_out = int(self.h_out), int(self.w_out)
        
        # 声明Img2colIndices实例
        self.img2col_indices = Img2colIndices((self.size, self.size), padding=0, stride=self.stride) # maxpool不需要padding
        
        return self.forward(X)
    
    def forward(self, X):
        """
        parameters:
            X: 输入feature，shape=(batch_size, channels, height, width)
        """
        x_reshaped = X.reshape(self.n_x * self.c_x, 1, self.h_x, self.w_x)
        self.x_col = self.img2col_indices.img2col(x_reshaped)
        self.max_indices = np.argmax(self.x_col, axis=0)
        
        out = self.x_col[self.max_indices, range(self.max_indices.size)]
        out = out.reshape(self.h_out, self.w_out, self.n_x, self.c_x).transpose(2, 3, 0, 1)
        return out
    
    def backward(self, d_out):
        """
        parameters:
            d_out: loss多maxpool输出的梯度，shape=(batch_size, channels, h_out, w_out)
        """
        d_x_col = np.zeros_like(self.x_col)  # shape=(size*size, h_out*h_out*batch*C)
        d_out_flat = d_out.transpose(2, 3, 0, 1).ravel()
        
        d_x_col[self.max_indices, range(self.max_indices.size)] = d_out_flat
        # 将d_x由col形态转换到img形态
        d_x = self.img2col_indices.col2img(d_x_col)
        d_x = d_x.reshape(self.n_x, self.c_x, self.h_x, self.w_x)
        
        return d_x


# 卷积网络辅助类img和col的转换
class Img2colIndices():
    """
    卷积网络的滑动计算实际上是将feature map转换成为矩阵乘法的方式。
    卷积计算forward前需要将feature map转换成为cols格式，每一次滑动的窗口作为cols的一列
    卷积计算backward时需要将cols态的梯度转换成为与输入map shape一致的格式
    该辅助类完成feature map --> cols 以及 cols --> feature map

    设计卷积、maxpool、average pool都有可能用到该类进行转换操作
    """
    def __init__(self, filter_size, padding, stride):
        """
        parameters:
            filter_shape: 卷积核的尺寸(h_filter, w_filter)
            padding: feature边缘填充0的个数
            stride: filter滑动步幅
        """
        self.h_filter, self.w_filter = filter_size
        self.padding = padding
        self.stride = stride
    
    def get_img2col_indices(self, h_out, w_out):
        """
        获得需要由image转换为col的索引, 返回的索引是在feature map填充后对于尺寸的索引

        获得每次卷积时，在feature map上卷积的元素的坐标索引。以后img2col时根据索引获得
        i 的每一行，如第r行是filter第r个元素(左右上下的顺序)在不同位置卷积时点乘的元素的位置的row坐标索引
        j 的每一行，如第r行是filter第r个元素(左右上下的顺序)在不同位置卷积时点乘的元素的位置的column坐标索引
        结果i、j每一列，如第c列是filter第c次卷积的位置卷积的k×k个元素(左右上下的顺序)。
        每一列长filter_height*filter_width*C，由于C个通道，每C个都是重复的，表示在第几个通道上做的卷积。

        parameters:
            h_out: 卷积层输出feature的height
            w_out: 卷积层输出feature的width。每次调用imgcol时计算得到
        return:
            k: shape=(filter_height*filter_width*C, 1), 每挨着的filter_height*filter_width元素值都一样，表示从第几个通道取点
            i: shape=(filter_height*filter_width*C, out_height*out_width), 依次待取元素的横坐标索引
            j: shape=(filter_height*filter_width*C, out_height*out_width), 依次待取元素的纵坐标索引
        """
        i0 = np.repeat(np.arange(self.h_filter), self.w_filter)
        i1 = np.repeat(np.arange(h_out), w_out) * self.stride
        i = i0.reshape(-1, 1) + i1
        i = np.tile(i, [self.c_x, 1])
        
        j0 = np.tile(np.arange(self.w_filter), self.h_filter)
        j1 = np.tile(np.arange(w_out), h_out) * self.stride
        j = j0.reshape(-1, 1) + j1
        j = np.tile(j, [self.c_x, 1])
        
        k = np.repeat(np.arange(self.c_x), self.h_filter * self.w_filter).reshape(-1, 1)
        
        return k, i, j
    
    def img2col(self, X):
        """
        基于索引取元素的方法实现img2col
        parameters:
            x: 输入feature map，shape=(batch_size, channels, height, width)
        return:
            转换img2col,shape=(h_filter * w_filter*chanels, batch_size * h_out * w_out)
        """
        self.n_x, self.c_x, self.h_x, self.w_x = X.shape

        # 首先计算出输出特征的尺寸
        # 计算输出feature的尺寸,并且保证是整数
        h_out = (self.h_x + 2 * self.padding - self.h_filter) / self.stride + 1
        w_out = (self.w_x + 2 * self.padding - self.w_filter) / self.stride + 1
        if not h_out.is_integer() or not w_out.is_integer():
            raise Exception("Invalid dimention")
        else:
            h_out, w_out = int(h_out), int(w_out)  # 上一步在进行除法后类型会是float
        
        # 0填充输入feature map
        x_padded = None
        if self.padding > 0:
            x_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            x_padded = X
        
        # 在计算出输出feature尺寸后,并且0填充X后，获得img2col_indices
        # img2col_indices设为实例的属性，col2img时用，避免重复计算
        self.img2col_indices = self.get_img2col_indices(h_out, w_out)
        k, i, j = self.img2col_indices
        
        # 获得参与卷积计算的col形式
        cols = x_padded[:, k, i, j]  # shape=(batch_size, h_filter*w_filter*n_channel, h_out*w_out)
        cols = cols.transpose(1, 2, 0).reshape(self.h_filter * self.w_filter * self.c_x, -1)  # reshape
        
        return cols
    
    def col2img(self, cols):
        """
        img2col的逆过程
        卷积网络，在求出x的梯度时，dx是col矩阵的形式(filter_height*filter_width*chanels, batch_size*out_height*out_width)
        将dx有col格式转换成feature map的原尺寸格式。由get_img2col_indices获得该尺寸下的索引，使用numpt.add.at方法还原成img格式
        parameters:
            cols: dx的col形式, shape=(h_filter*w_filter*n_chanels, batch_size*h_out*w_out)
        """
        # 将col还原成img2col的输出shape
        cols = cols.reshape(self.h_filter * self.w_filter * self.c_x, -1, self.n_x)
        cols = cols.transpose(2, 0, 1)
        
        h_padded, w_padded = self.h_x + 2 * self.padding, self.w_x + 2 * self.padding
        x_padded = np.zeros((self.n_x, self.c_x, h_padded, w_padded))
        
        k, i, j = self.img2col_indices
        
        np.add.at(x_padded, (slice(None), k, i, j), cols)
        
        if self.padding == 0:
            return x_padded
        else:
            return x_padded[:, :, self.padding : -self.padding, self.padding : -self.padding]


# 定义BatchNorm2d
class BatchNorm2d():
    """
    对卷积层来说，批量归一化发生在卷积计算之后、应用激活函数之前。
    如果卷积计算输出多个通道，我们需要对这些通道的输出分别做批量归一化,且每个通道都拥有独立的拉伸和偏移参数，并均为标量。
    设小批量中有 m 个样本。在单个通道上，假设卷积计算输出的高和宽分别为 p 和 q 。我们需要对该通道中 m×p×q 个元素同时做批量归一化。
    对这些元素做标准化计算时，我们使用相同的均值和方差，即该通道中 m×p×q 个元素的均值和方差。
    
    将训练好的模型用于预测时，我们希望模型对于任意输入都有确定的输出。
    因此，单个样本的输出不应取决于批量归一化所需要的随机小批量中的均值和方差。
    一种常用的方法是通过移动平均估算整个训练数据集的样本均值和方差，并在预测时使用它们得到确定的输出。
    """
    def __init__(self, n_channel, momentum):
        """
        parameters:
            n_channel: 输入feature的通道数
            momentum: moving_mean/moving_var迭代调整系数
        """
        self.n_channel = n_channel
        self.momentum = momentum
        
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = np.ones((1, n_channel, 1, 1))
        self.beta = np.zeros((1, n_channel, 1, 1))
        
        # 测试时使用的参数，初始化为0，需在训练时动态调整
        self.moving_mean = np.zeros((1, n_channel, 1, 1))
        self.moving_var = np.zeros((1, n_channel, 1, 1))
        
        self.params = [self.gamma, self.beta]
    
    def __call__(self, X, mode):
        """
        X: shape = (N, C, H, W)
        mode: 训练阶段还是测试阶段，train或test, 需要在调用时传参
        """
        self.X = X  # 求gamma的梯度时用
        return self.forward(X, mode)
    
    def forward(self, X, mode):
        """
        X: shape = (N, C, H, W)
        mode: 训练阶段还是测试阶段，train或test, 需要在调用时传参
        """
        if mode != 'train':
            # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
            self.x_norm = (X - self.moving_mean) / np.sqrt(self.moving_var + 1e-5)
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。
            # 这里我们需要保持X的形状以便后面可以做广播运算
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            self.var = X.var(axis=(0, 2, 3), keepdims=True)  # 设为self，是因为backward时会用到
            
            # 训练模式下用当前的均值和方差做标准化。设为类实例的属性，backward时用
            self.x_norm = (X - mean) / (np.sqrt(self.var + 1e-5))
            
            # 更新移动平均的均值和方差
            self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * mean
            self.moving_var = self.momentum * self.moving_var + (1 - self.momentum) * self.var
        # 拉伸和偏移
        out = self.x_norm * self.gamma + self.beta
        return out
    
    def backward(self, d_out):
        """
        d_out的形状与输入的形状一样
        """
        d_gamma = (d_out * self.x_norm).sum(axis=(0, 2, 3), keepdims=True)
        d_beta = d_out.sum(axis=(0, 2, 3), keepdims=True)
        
        d_x = (d_out * self.gamma) / np.sqrt(self.var + 1e-5)
        
        return d_x, [d_gamma, d_beta]


# 定义Flatten，卷积层到全连接层
class Flatten():
    """
    最后的卷积层输出的feature若要连接全连接层需要将feature拉平
    单独建立一个模块是为了方便梯度反向传播
    """
    def __init__(self):
        pass
    
    def __call__(self, X):
        self.x_shape = X.shape # (batch_size, channels, height, width)
        
        return self.forward(X)
    
    def forward(self, X):
        out = X.ravel().reshape(self.x_shape[0], -1)
        return out
    
    def backward(self, d_out):
        d_x = d_out.reshape(self.x_shape)
        return d_x