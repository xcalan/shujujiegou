class Node:
    def __init__(self, elem):
        self.elem = elem
        self.next = None


class SingleLinkList:
    def __init__(self, node=None):
        self._head = node

    # 判断是否为为空
    def is_empty(self):
        return self._head is None

    # 链表长度
    def length(self):
        cur = self._head # 游标
        count = 0
        while cur!=None:
            count += 1
            cur = cur.next
        return count

    # 遍历
    def travel(self):
        cur = self._head
        while cur!=None:
            print(cur.elem, end=" ")
            cur = cur.next
        print("")

    # 头部添加元素
    def add(self, item):
        node = Node(item)
        node.next = self._head
        self._head = node

    # 尾部添加元素
    def append(self, item):
        cur = self._head
        node = Node(item)
        if cur==None:
            self._head=node
        else:
            while cur.next != None:
                cur = cur.next
            cur.next = node

    # 指定位置添加元素
    def insert(self, pos, item):
        node = Node(item)
        pre = self._head
        count = 0
        if pos <= 0:
            self.add(item)
        elif pos > self.length()-1:
            self.append(item)
        else:
            while count < pos-1:
                pre = pre.next
                count += 1
            node.next = pre.next
            pre.next = node

    # 查找
    def search(self, item):
        cur = self._head
        while cur != None:
            if cur.elem == item:
                return True
            else:
                cur = cur.next
        return False

    # 删除节点
    def remove(self, item):
        cur = self._head
        pre = None
        while cur != None:
            if cur.elem == item:
                if cur == self._head: #如果是首结点
                    self._head = cur.next
                else:
                    pre.next = cur.next
                break
            else:
                pre = cur
                cur = cur.next



if __name__ == '__main__':
    node1 = Node(1)
    sll = SingleLinkList(node1)
    sll.append(2)
    sll.append(3)
    sll.append(4)
    sll.append(5)
    sll.travel()

    sll.insert(-1, 0)
    sll.travel()#012345
    sll.insert(2, 1.5)
    sll.travel()#01 1.5 2345
    sll.insert(10, 6)
    sll.insert(10, 6)
    sll.travel()  #0 1 1.5 2 3 4 5 6
    sll.remove(6)
    sll.travel()