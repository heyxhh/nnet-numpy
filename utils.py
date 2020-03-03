import numpy as np

# 数字label与one hot转换
def label_encoder(label, num_class):
    """
    将class有0，1，。。。，n-1编码为one-hot
    label: 原标签，值为0，1，2，。。。，num_class-1
    num_class: 标签的种类数
    return: 没一行是一个sample的标签的one-hot, shape=(m, num_class)
    """
    tmp = np.eye(num_class)
    return tmp[label]

# one hot转化为数字
def label_decoder(one_hot):
    """
    将one-hot转换成数值
    one_hot：one hot形式的label。array矩阵，shape=(m, num_class)
    return：数值型的label，shape=(n)
    """
    return np.argmax(one_hot, axis=1)

# 打乱数据的顺序
def shuffle_data(datas, labels):
    """
    随机打乱数据顺序
    参数：
        datas, labels
    """
    n = labels.shape[0]  # 总数据量
    # 打乱顺序
    shuffled_idx = np.arange(n)
    np.random.shuffle(shuffled_idx)
    shuffled_datas, shuffled_labels = datas[shuffled_idx], labels[shuffled_idx]
    return shuffled_datas, shuffled_labels


def softmax(X):
    """
    根据X，计算softmax
    代码利用softmax函数的性质: softmax(x) = softmax(x + c)
    return: softmax
    """
    batch_size = X.shape[0]
    # axis=1 表示在二维数组中沿着横轴进行取最大值的操作
    max_value = X.max(axis=1)
    #每一行减去自己本行最大的数字,防止取指数后出现inf，性质：softmax(x) = softmax(x + c)
    tmp = X - max_value.reshape(batch_size, 1)
    # 对每个数取指数
    exp_input = np.exp(tmp)  # shape=(m, n)
    # 求出每一行的和
    exp_sum = exp_input.sum(axis=1).reshape(batch_size, 1)  # shape=(m, 1)
    return exp_input / exp_sum