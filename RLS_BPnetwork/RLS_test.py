# -*- coding:utf-8 -*-

# python 实现递推最小二乘辨识
# author by：Xie Cheng
# data：2019.5.10

import numpy as np
from numpy import eye, mat, zeros
import math
import matplotlib.pyplot as plt
import csv


# 绝对值的平均值函数
def mean_value(alist):
    sum = 0
    n = len(alist)
    for item in alist:
        sum += abs(item)
    return sum/n


# 归一化函数（没用上）
def norm_trans(list):
    list_np = np.asarray(list)
    result = []
    for item in list_np:
        # item = float(item - np.min(list_np)) / (np.max(list_np) - np.min(list_np))
        result.append(item)
    return mat(result)


# 均方根误差函数
def RMSE_calu(list1, list2):
    sum = 0
    for i in range(len(list1)):
        num = pow(list1[i]-list2[i], 2)
        sum += num
    res = math.sqrt(sum/len(list1))
    return res


# 参数训练
def RLS_test(u_list, y_list, F_init, Q_init, delta_y_init):

    a_i_list = []
    a_action_list = []

    # 频率
    for item in u_list:
        a_action_list.append(item)

    # 电流
    for item in y_list:
        a_i_list.append(item)

    a_i_list = [a_i_list[i] - delta_y_init[i] for i in range(len(delta_y_init))] # y-△y 的list
    # 得到u和y-△y的两个list

    L = len(a_i_list)# 数据总数
    # print(L)

    # 归一化（此处没归一化）,转换矩阵形式,[[....]]
    a_i = norm_trans(a_i_list)
    a_action = norm_trans(a_action_list)

    P1 = mat(pow(10, 6) * eye(2))
    FQ1_e = mat(zeros((2, L)))
    # FQ1_e0 = mat([[5.27155118920296e-07], [2.42058865913553e-07]])# 给F、Q赋初值
    FQ1_e0 = mat([[F_init], [Q_init]])  # 给F、Q赋初值
    F1_tr = 0.0
    Q1_tr = 0.0

    # 训练参数过程
    for k in range(0, L-1):
        phi1 = mat([[(math.sqrt(3)) / math.pi * pow(a_i[0, k], 2)], [-2*math.sqrt(3)*a_action[0, k]*pow(a_i[0, k], 2)]])
        K1 = P1*phi1/(1+phi1.T*P1*phi1)
        FQ1_e[0:, k:k+1] = FQ1_e0+K1*((a_i[0, k+1]-a_i[0, k])-phi1.T*FQ1_e0)
        P1 = (eye(2)-K1*phi1.T)*P1
        if np.linalg.norm(FQ1_e[0:, k:k+1]-FQ1_e0) > 5*np.linalg.norm(FQ1_e0):
            FQ1_e[0:, k:k + 1] = FQ1_e0
        FQ1_e0 = FQ1_e[0:, k:k + 1]
        F1_tr = FQ1_e0[0, 0]
        Q1_tr = FQ1_e0[1, 0]
        # print("第%d次训练: F1=%s, Q1=%s" % (k, F1_tr, Q1_tr))
    print("RLS训练结果：F: %s Q: %s" % (F1_tr, Q1_tr))

    return F1_tr, Q1_tr


    # ###################### 验证过程 ###############################
    # csvFile_test = open("vali.csv", "r")
    # reader_test = csv.reader(csvFile_test)
    #
    # a_i_vali_list = []
    # b_i_vali_list = []
    # c_i_vali_list = []
    # a_action_vali_list = []
    # b_action_vali_list = []
    # c_action_vali_list = []
    #
    # # 用于验证的数据从CSV文件中导入
    # for item in reader_test:
    #     a_i_vali_list.append(float(item[0]))
    #     b_i_vali_list.append(float(item[1]))
    #     c_i_vali_list.append(float(item[2]))
    #     a_action_vali_list.append(float(item[3]))
    #     b_action_vali_list.append(float(item[4]))
    #     c_action_vali_list.append(float(item[5]))
    # csvFile.close()
    #
    # # 用于验证的数据个数
    # L_vali_num = len(a_i_vali_list)
    #
    # # 根据电流模型算出的输出值
    # y1_kp1_val = zeros((1, L_vali_num-1))
    #
    # a_i_vali = mat(a_i_vali_list)
    # a_action_vali = mat(a_action_vali_list)
    #
    # for k in range(0, L_vali_num-1):
    #     phi1_test = mat([[(math.sqrt(3)) / math.pi * pow(a_i_vali[0, k], 2)], [-2*math.sqrt(3)*a_action_vali[0, k]*pow(a_i_vali[0, k], 2)]])
    #     theta1_test = mat([[F1_tr], [Q1_tr]])
    #     y1_kp1_val[0, k] = phi1_test.T*theta1_test+a_i_vali[0, k]
    #     print("第%d个电流值：%s " % (k+1, float(y1_kp1_val[0, k])))
    #
    # y1_model = y1_kp1_val.tolist()[0]
    #
    # # 画图
    # plt.figure()
    # t_len = len(y1_model)
    # t = list(range(t_len))
    # plt.plot(t, y1_model, '-', t, a_i_vali_list[0:t_len], '-')
    # plt.show()
    #
    # rmse_result = RMSE_calu(y1_model, a_i_vali_list[0:t_len])
    # print("均方根误差：%s" % (rmse_result))


# if __name__ == '__main__':
#     csvFile = open("FMF_data.csv", "r")
#     reader_test = csv.reader(csvFile)
#     RLS_test(reader_test)
