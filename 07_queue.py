# coding:utf-8


# 队列
class Queue:
    def __init__(self):
        self.__list = []

    def enqueue(self,item):
        # 往队列里添加item元素
        self.__list.append(item)# 1.尾部添加
        # self.__list.insert(0, item)# 2.头部添加

    def dequeue(self):
        # 从队列头部删除一个元素
        return self.__list.pop(0)# 1.头部弹出
        # return self.__list.pop()# 2.尾部弹出

    def is_empty(self):
        # 判断一个队列是否为空
        return self.__list == []

    def size(self):
        # 返回队列大小
        return len(self.__list)

if __name__ == '__main__':
    q = Queue()
    q.enqueue(1)
    q.enqueue(2)
    q.enqueue(3)
    q.enqueue(4)
    print(q.dequeue())
    print(q.dequeue())
    print(q.dequeue())
    print(q.dequeue())