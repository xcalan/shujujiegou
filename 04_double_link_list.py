class Node:
    def __init__(self, elem):
        self.elem = elem
        self.next = None
        self.prev = None

class DoubleLinkList:
    def __init__(self, node=None):
        self._head = node






if __name__ == '__main__':
    node1 = Node(1)
    sll = DoubleLinkList(node1)
