class Node:
    def __init__(self, elem):
        self.elem = elem
        self.next = None
        self.prev = None


class DoubleLinkList:
    def __init__(self, node=None):
        self.__head = node

    def is_empty(self):
        # 链表是否为空
        return self.__head is None

    def length(self):
        # 链表长度
        cur = self.__head
        count = 0
        while cur != None:
            count += 1
            cur = cur.next
        return count

    def travel(self):
        # 遍历
        cur = self.__head
        while cur != None:
            print(cur.elem, end=" ")
            cur = cur.next
        return

    def add(self, item):
        # 链表头部添加
        node = Node(item)
        if self.__head is None:
            self.__head = node
        else:
            cur = self.__head
            cur.prev = node
            node.next = cur
            self.__head = node

    def append(self, item):
        # 链表尾部添加
        node = Node(item)
        if self.__head is None:
            self.__head = node
        else:
            cur = self.__head
            while cur.next is not None:
                cur = cur.next
            cur.next = node
            node.prev = cur

    def insert(self, pos, item):
        # 指定位置添加
        node = Node(item)
        count = 0
        cur = self.__head
        if pos <= 0:
            self.add(item)
        elif pos > (self.length()-1):
            self.append(item)
        else:
            while count < pos:
                count += 1
                cur = cur.next
            cur.prev.next = node
            node.prev = cur.prev
            node.next = cur
            cur.prev = node

    def remove(self, item):
        # 删除节点
        cur = self.__head
        if cur is None:
            return
        while cur != None:
            if cur.elem == item:
                if cur.prev is None:
                    self.__head = cur.next
                    cur.next.prev = None
                    cur.next = None
                elif cur.next is None:
                    cur.prev.next = None
                    cur.prev = None
                else:
                    cur.prev.next = cur.next
                    cur.next.prev = cur.prev
                    cur.next = None
                    cur.prev = None
            cur = cur.next

    def search(self, item):
        # 查找节点是否存在
        cur = self.__head
        while cur != None:
            if cur.elem == item:
                return True
            cur = cur.next
        return False


if __name__ == '__main__':
    node1 = Node(0)
    sll = DoubleLinkList(node1)
    sll.append(1)
    sll.append(2)
    sll.append(3)
    sll.append(4)
    sll.append(5)
    sll.travel()
    print(" ")
    # print(sll.length())
    sll.insert(2, 1.5)
    sll.travel()
    print(" ")
    sll.remove(5)
    sll.travel()
