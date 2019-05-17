# coding:utf-8
# 堆排序


def heapify(tree, n, i):
    # n = len(tree)
    if i > n-1:
        return
    c1 = 2*i+1
    c2 = 2*i+2
    max = i
    if c1 < n:
        if tree[c1] > tree[max]:
            max = c1

    if c2 < n:
        if tree[c2] > tree[max]:
            max = c2

    if max != i:
        tree[i], tree[max] = tree[max], tree[i]
        heapify(tree, n, max)


# 得到一个大根堆
def build_heap(tree):
    n = len(tree)
    last_node = n-1 # 最后一个叶子节点的下标
    parent = (last_node-1)//2 # 最后一个叶子结点的父节点
    for i in range(parent, -1, -1):
       heapify(tree, n, i)


def heap_sort(tree):
    n = len(tree)
    build_heap(tree)
    for i in range(n-1, -1, -1):
        tree[i], tree[0] = tree[0], tree[i]
        heapify(tree, i, 0)


if __name__ == '__main__':
    tree = [4, 10, 3, 5, 1, 2]
    # heapify(tree, 0)
    heap_sort(tree)
    print(tree)