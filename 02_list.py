from timeit import Timer


def t1():
    li=[]
    for i in range(10000):
        li.append(i)

def t2():
    li=[]
    for i in range(10000):
        li = li+[i]

def t3():
    li = [i for i in range(10000)]

def t4():
    li = list(range(10000))

def t5():
    li=[]
    for i in range(10000):
        li.extend(i)


'''
    第一个参数表示函数代码段
    第二个参数表示从main中引入对应测试函数
'''
timer1=Timer("t2()", "from __main__ import t2")
print(timer1.timeit(1))#执行次数


