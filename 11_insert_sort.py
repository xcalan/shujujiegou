# coding:utf-8
# 插入排序


def insert_sort(alist):
    n = len(alist)
    for i in range(1, n):
        for j in range(i, 0, -1):
            if alist[j] < alist[j-1]:
                alist[j], alist[j-1] = alist[j-1], alist[j]
            else:
                break
    return alist


if __name__ == '__main__':
    l = [54, 26, 93, 17, 77, 31, 44, 55, 20]
    print(insert_sort(l))