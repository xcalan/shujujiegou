# coding:utf-8
# 二叉树


class Node:
    def __init__(self,item):
        self.elem = item
        self.lchild = None
        self.rchild = None


class Tree:
    """二叉树"""
    def __init__(self):
        self.root = None

    def add(self, item):
        node = Node(item)
        if self.root is None:
            self.root = node
            return
        queue = [self.root]
        while queue:
            cur_node = queue.pop(0)
            if cur_node.lchild is None:
                cur_node.lchild = node
                return
            else:
                queue.append(cur_node.lchild)
            if cur_node.rchild is None:
                cur_node.rchild = node
                return
            else:
                queue.append(cur_node.rchild)

    def breadth_travel(self):
        """广度遍历"""
        if self.root is None:
            return
        queue = [self.root]
        while queue: # list判空条件
            cur_node = queue.pop(0)
            print(cur_node.elem, end=" ")
            if cur_node.lchild is not None:
                queue.append(cur_node.lchild)
            if cur_node.rchild is not None:
                queue.append(cur_node.rchild)

    def preorder(self, node):
        """先序遍历"""
        if node is None:
            return
        print(node.elem, end=" ")
        self.preorder(node.lchild)
        self.preorder(node.rchild)

    def inorder(self, node):
        """中序遍历"""
        if node is None:
            return
        self.inorder(node.lchild)
        print(node.elem, end=" ")
        self.inorder(node.rchild)

    def postorder(self, node):
        """后序遍历"""
        if node is None:
            return
        self.postorder(node.lchild)
        self.postorder(node.rchild)
        print(node.elem, end=" ")


if __name__ == '__main__':
    tr = Tree()
    tr.add(0)
    tr.add(1)
    tr.add(2)
    tr.add(3)
    tr.add(4)
    tr.add(5)
    tr.add(6)
    tr.add(7)
    tr.add(8)
    tr.add(9)

    tr.breadth_travel()
    print(" ")
    tr.preorder(tr.root)
    print(" ")
    tr.inorder(tr.root)
    print(" ")
    tr.postorder(tr.root)
