# coding:utf-8
# 选择排序


def select_sort(alist):
    n = len(alist)
    for i in range(n-1):
        min_index = i
        for j in range(i+1, n):
            if alist[min_index] > alist[j]:
                min_index = j
        alist[i], alist[min_index] = alist[min_index], alist[i]
    return alist


if __name__ == '__main__':
    l = [54, 26, 93, 17, 77, 31, 44, 55, 20]
    print(select_sort(l))