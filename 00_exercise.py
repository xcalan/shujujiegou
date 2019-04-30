class Node:
    def __init__(self, elem):
        self.elem = elem
        self.next = None


class SingleList:
    def __init__(self, node=None):
        self._head = node

    # 尾插
    def append(self, item):
        node = Node(item)
        cur = self._head
        if self._head == None:
            self._head = node
        else:
            while cur.next != None:
                cur = cur.next
            cur.next = node

    # 遍历
    def travel(self):
        cur = self._head
        while cur != None:
            print(cur.elem, end=" ")
            cur = cur.next

    def printListFromTailToHead(self):
        cur = self._head
        l=[]
        while cur != None:
            l += [cur.elem]
            cur=cur.next
        return l[::-1]

if __name__ == '__main__':
    sll=SingleList()
    sll.append(1)
    sll.append(2)
    sll.append(3)
    sll.append(4)
    sll.append(5)
    sll.travel()
    print(" ")
    arr = sll.printListFromTailToHead()
    print(arr)
