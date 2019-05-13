# -*- coding:utf-8 -*-

# 主函数，实现递推最小二乘+神经网络交替辨识
# author by：Xie Cheng
# data：2019.5.12

from numpy import *
from RLS_BPnetwork.RLS_test import *
import datetime
import matplotlib.pyplot as plt

"""
导入用于校正的数据，文件格式为CSV格式
"""
# 此处应添加本地选择完离线数据后，下载对应的CSV文件，并发送到云端指定的文件夹内

# CSV文件读取
csvFile_test = open("FMF_data.csv", "r")
reader_test = csv.reader(csvFile_test)

# A、B、C初始化
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

# # 如果对B相训练，执行
# a_i = b_i
# a_action = b_action
#
# # 如果对C相训练，执行
# a_i = c_i
# a_action = c_action


# 数据个数
L_num = len(a_i)

# F、Q、△y初值
F1 = 5.27155118920296e-07
Q1 = 2.42058865913553e-07
delta_y_init = [0]*(L_num-1)

# 真实电流值list转换成真实电流值矩阵
a_i_vali = mat(a_i)
# 真实频率值list转换成真实频率值矩阵
a_action_vali = mat(a_action)

u_rls = a_action
y_rls = a_i

"""
定义BP神经网络
"""
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


"""
循环以下过程，直至停止条件结束
"""
print("开始时间: %s" % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
while True:
    """
    递推最小二乘辨识
    """
    F1, Q1 = RLS_test(u_rls, y_rls, F1, Q1, delta_y_init)

    """
    网络训练前序工作，得到训练集
    """
    # 电流模型输出值(初始化)
    y1_kp1_val = zeros((1, L_num-1))

    # 根据电流模型得到模型输出值（+△y）
    for k in range(0, L_num-1):
        phi1_test = mat([[(math.sqrt(3)) / math.pi * pow(a_i_vali[0, k], 2)], [-2*math.sqrt(3)*a_action_vali[0, k]*pow(a_i_vali[0, k], 2)]])
        theta1_test = mat([[F1], [Q1]])
        y1_kp1_val[0, k] = phi1_test.T*theta1_test+a_i_vali[0, k]+delta_y_init[k]
        # print("第%d个电流值：%s " % (k+1, float(y1_kp1_val[0, k])))

    y1_model = y1_kp1_val.tolist()[0] # 是一个装有模型输出电流值的list
    y1_real = a_i[0:L_num-1] # 是一个装有真实电流值的list
    # print(y1_model)
    delta_y = [y1_real[i] - y1_model[i] for i in range(L_num-1)] # 这是一个装有△y的list
    # print(delta_y)
    print("△y的均值为:%s" % mean_value(delta_y))

    # 判断是否达到停止条件
    if mean_value(delta_y) < 100:
        break
    else:
        # 根据电流模型对△y进行处理
        for k in range(0, L_num-1):
            delta_y[k] = (delta_y[k]/pow(a_i[k], 2)) * pow(10, 7)
        # print(delta_y)                # △y:list形式
        u = a_action[0:len(a_action)-1] # u:list形式
        # print(u)

        """
        BP神经网络训练
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
            delta_y.append(delta_y_pred[k]*pow(a_i[k], 2)/pow(10, 7))
        # print(delta_y)
        delta_y_init = delta_y # 更新△y，交给递推最小二乘算法
        print("BP神经网络训练的△y:%s" % delta_y)

# 获得电熔镁炉模型   F、Q 为递推最小二乘训练的结果，分别为一个数，delta_y_init为一个list
print("最终训练结果为：F:%s Q:%s △y(最后一项):%s" % (F1, Q1, delta_y_init[-1]))
print("结束时间: %s" % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


"""
画图：真实电流值和模型值对比
"""

# 数据设置
t_len = len(y1_model)
t = list(range(t_len))

# 模型值
x1 = t
y1 = y1_model[0:t_len]

# 实际值
x2 = t
y2 = y1_real

# 设置输出的图片大小
figsize = 25, 10
figure, ax = plt.subplots(figsize=figsize)

# 在同一幅图片上画两条折线
A, = plt.plot(x1, y1, '-r', label='y_model', linewidth=3.0)
B, = plt.plot(x2, y2, '-.b', label='y_real', linewidth=2.0)

# 设置图例并且设置图例的字体及大小
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 23,
         }
legend = plt.legend(handles=[A, B], prop=font1)

# 设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=23)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

# 设置横纵坐标的名称以及对应字体格式
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 30,
         }
plt.xlabel('num', font2)
plt.ylabel('y(A)', font2)

# 将文件保存至文件中并且画出图
plt.savefig('figure.eps')
plt.show()


