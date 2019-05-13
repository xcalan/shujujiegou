# -*- coding:utf-8 -*-

# python 实现BP神经网络
# author by：Xie Cheng
# data：2019.5.10

import numpy as np
import math
import matplotlib.pyplot as plt
import csv
from numpy import mat, zeros


class ApproachNetwork:
    """
    输入层和隐藏层之间有权重W1和偏置B1，隐藏层与输出层之间有权重W2和和偏置B2
    """
    def __init__(self, hidden_size=100, output_size=1):
        self.learning_rate = 0.05  # 定义学习率
        self.params = {'W1': np.random.random((1, hidden_size)),
                       'B1': np.zeros(hidden_size),
                       'W2': np.random.random((hidden_size, output_size)),
                       'B2': np.zeros(output_size)}

    # 定义sigmoid函数
    @staticmethod
    def sigmoid(x_):
        return 1 / (1 + np.exp(-x_))

    def sigmoid_grad(self, x_):
        return (1.0 - self.sigmoid(x_)) * self.sigmoid(x_)

    # 前馈网络输出
    def predict(self, x_):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['B1'], self.params['B2']

        a1 = np.dot(x_, W1) + b1
        z1 = self.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2

        return a2

    # 损失函数
    def loss(self, x_, t):
        y_ = self.predict(x_)
        return y_, np.mean((t - y_) ** 2)

    # 计算BP网络梯度和更新权重
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['B1'], self.params['B2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = self.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2

        # backward
        dy = (a2 - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['B2'] = np.sum(dy, axis=0)

        dz1 = np.dot(dy, W2.T)
        da1 = self.sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['B1'] = np.sum(da1, axis=0)

        return grads

    def train_with_own(self, x_, y_, max_steps=100):
        for k in range(max_steps):
            grad = self.gradient(x_, y_)
            for key in ('W1', 'B1', 'W2', 'B2'):
                self.params[key] -= self.learning_rate * grad[key]
            pred, loss = network.loss(x_, y_)

            """
            画图过程
            """
            # 动态绘制结果图，每训练100次一变，可以看到训练过程如何慢慢的拟合数据点
            if k % 100 == 0:
                plt.cla()
                plt.plot(x, y, lw=3)
                plt.plot(x, pred, 'r-', lw=3)
                plt.text(0.5, 0, 'Loss=%.4f' % abs(loss), fontdict={'size': 20, 'color': 'red'})
                plt.pause(0.1)

        # 关闭动态绘制模式
        plt.ioff()
        plt.show()
        l_pred = []
        for i in range(len(pred)):
            l_pred.append(pred[i][0])
        return l_pred


if __name__ == '__main__':

    csvFile_test = open("FMF_data.csv", "r")
    reader_test = csv.reader(csvFile_test)

    a_i = []
    b_i = []
    c_i = []
    a_action = []
    b_action = []
    c_action = []

    # 数据从CSV文件中导入
    for item in reader_test:
        a_i.append(float(item[0]))
        b_i.append(float(item[1]))
        c_i.append(float(item[2]))
        a_action.append(float(item[3]))
        b_action.append(float(item[4]))
        c_action.append(float(item[5]))
    csvFile_test.close()

    # 数据个数
    L_num = len(a_i)

    # 真实电流值list转换成真实电流值矩阵
    a_i_vali = mat(a_i)
    # 真实频率值list转换成真实频率值矩阵
    a_action_vali = mat(a_action)
    """
    从此处循环...
    """

    # 从递推最小二乘算法程序导入F、Q
    F1 = 5.27155118920296e-07
    Q1 = 2.42058865913553e-07

    # 电流模型输出值(初始化)
    y1_kp1_val = zeros((1, L_num-1))

    # 根据电流模型得到模型输出值
    for k in range(0, L_num-1):
        phi1_test = mat([[(math.sqrt(3)) / math.pi * pow(a_i_vali[0, k], 2)], [-2*math.sqrt(3)*a_action_vali[0, k]*pow(a_i_vali[0, k], 2)]])
        theta1_test = mat([[F1], [Q1]])
        y1_kp1_val[0, k] = phi1_test.T*theta1_test+a_i_vali[0, k]
        # print("第%d个电流值：%s " % (k+1, float(y1_kp1_val[0, k])))

    y1_model = y1_kp1_val.tolist()[0] # 是一个装有模型输出电流值的list
    y1_real = a_i[0:L_num-1] # 是一个装有真实电流值的list
    # print(y1_model)
    delta_y = [y1_real[i] - y1_model[i] for i in range(L_num-1)]
    # print(delta_y)
    # 根据电流模型对△y进行处理
    for k in range(0, L_num-1):
        delta_y[k] = (delta_y[k]/pow(a_i[k], 2)) * pow(10, 7)
    # print(delta_y)                # △y:list形式
    u = a_action[0:len(a_action)-1] # u:list形式
    # print(u)

    """
    BP神经网络训练过程
    """
    network = ApproachNetwork()
    # 训练集,结构如下：
    # x = np.array(mat([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]))
    # y = np.array(mat([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]))
    x = np.array(mat(u).T) # 训练集输入
    y = np.array(mat(delta_y).T) # 训练集输出

    # 使用 BP network 训练  x:训练集输入，y:训练集输出，4000为训练次数（可改）,delta_y_pred:模型预报值
    delta_y_pred = network.train_with_own(x, y, 4000)
    # print(delta_y_pred)
    # print(len(delta_y_pred))
    # 对△y_pred进行反处理，得到△y
    delta_y = []
    for k in range(0, len(delta_y_pred)):
        delta_y.append(delta_y_pred[k]*pow(y1_real[k], 2)/pow(10, 7))
    print(delta_y)
    # y1_real = [y1_real[i]-delta_y[i] for i in range(len(delta_y_pred))]
    # # print(y1_real)
    # # 此处将y1_real和a_action交给递推最小二乘算法辨识参数
