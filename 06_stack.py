# coding:utf-8


# 栈
class Stack:
    def __init__(self):
        self.__list = []

    def push(self, item):
        # 添加一个元素到栈顶
        self.__list.append(item)

    def pop(self):
        # 弹出栈顶元素
        return self.__list.pop()

    def peek(self):
        # 返回栈顶元素
        if self.__list is None:
            return None
        return self.__list[-1]

    def is_empty(self):
        # 判断栈是否为空
        return self.__list == []

    def size(self):
        # 返回栈元素的个数
        return len(self.__list)


if __name__ == '__main__':
    s = Stack()
    print(s.is_empty())
    s.push(1)
    s.push(2)
    s.push(3)
    s.push(4)
    print(s.is_empty())
    print("栈顶元素:%s" % s.peek())
    print(s.size())
    print(s.pop())
    print(s.pop())
    print(s.pop())
    print(s.pop())

